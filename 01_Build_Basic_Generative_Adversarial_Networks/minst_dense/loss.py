
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
import torch

from device import device
from noise import get_noise


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    noise     = get_noise(num_images, z_dim)
    fake_img  = gen(noise).detach()
    fake_pred = disc(fake_img)
    real_pred = disc(real)
    loss_gen  = criterion(fake_pred, torch.zeros((num_images,1),).to(device))
    loss_real = criterion(real_pred, torch.ones( (num_images,1),).to(device))
    disc_loss = (loss_gen + loss_real) / 2.0
    return disc_loss




def get_gen_loss(gen, disc, criterion, num_images, z_dim):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    noise     = get_noise(num_images, z_dim).to(device)
    fake_img  = gen(noise)
    fake_pred = disc(fake_img)
    gen_loss  = criterion(fake_pred, torch.ones((num_images,1),).to(device))
    return gen_loss
