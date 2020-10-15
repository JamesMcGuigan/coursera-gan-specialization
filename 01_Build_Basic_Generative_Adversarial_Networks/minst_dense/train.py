import atexit
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST  # Training dataset

from device import device
from loss import get_disc_loss
from loss import get_gen_loss
from MinstDiscriminator import MinstDiscriminator
from MinstGenerator import MinstGenerator
from noise import get_noise
from plot import show_tensor_images


def train(z_dim=64, lr=0.001, n_epochs=200, display_step=500, batch_size=128):
    time_start = time.perf_counter()
    dataloader = DataLoader(
        MNIST('.', download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.BCEWithLogitsLoss()

    gen      = MinstGenerator(z_dim).to(device).load().train().unfreeze()
    gen_opt  = torch.optim.Adam(gen.parameters(), lr=lr)
    disc     = MinstDiscriminator().to(device).load().train().unfreeze()
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr/100)

    atexit.register(gen.save)
    atexit.register(disc.save)

    try:
        cur_step                = 0
        mean_generator_loss     = 0
        mean_discriminator_loss = 0
        test_generator          = False  # Whether the generator should be tested
        best_gen_loss = math.inf
        gen_loss  = math.inf
        disc_loss = math.inf
        error    = False
        for epoch in range(n_epochs):

            # Dataloader returns the batches
            # for real, _ in tqdm(dataloader):
            for real, _ in dataloader:
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.view(cur_batch_size, -1).to(device)

                ### Update discriminator ###
                # Zero out the gradients before backpropagation
                best_gen_loss = min(gen_loss, best_gen_loss)

                disc_opt.zero_grad()
                disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim)
                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                # For testing purposes, to keep track of the generator weights
                if test_generator:
                    old_generator_weights = gen.gen[0][0].weight.detach().clone()

                ### Update generator ###
                #     Hint: This code will look a lot like the discriminator updates!
                #     These are the steps you will need to complete:
                #       1) Zero out the gradients.
                #       2) Calculate the generator loss, assigning it to gen_loss.
                #       3) Backprop through the generator: update the gradients and optimizer.
                gen_opt.zero_grad()
                gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim)
                gen_loss.backward()
                gen_opt.step()


                # For testing purposes, to check that your code changes the generator weights
                if test_generator:
                    try:
                        assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                        assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                    except:
                        error = True
                        print("Runtime tests have failed")

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_loss.item() / display_step

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / display_step

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    print(f"Epoch {epoch:4d} | step {cur_step:6d} | Generator loss: {mean_generator_loss:19.16f} | discriminator loss: {mean_discriminator_loss:19.16f}")
                    fake_noise = get_noise(cur_batch_size, z_dim)
                    fake = gen(fake_noise)
                    show_tensor_images(fake)
                    # show_tensor_images(real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1

    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as exception:
        print(exception)
        raise exception
    finally:
        time_taken = time.perf_counter() - time_start
        gen.save(verbose=True)
        disc.save(verbose=True)
        atexit.unregister(gen.save)     # model now saved, so cancel atexit handler
        atexit.unregister(disc.save)    # model now saved, so cancel atexit handler

if __name__ == '__main__':
    train()
