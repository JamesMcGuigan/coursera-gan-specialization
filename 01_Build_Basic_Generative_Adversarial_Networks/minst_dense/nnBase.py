from __future__ import annotations

import math
import os
import re
from abc import ABCMeta
from typing import List
from typing import TypeVar
from typing import Union

import humanize
import numpy as np
import torch
import torch.nn as nn


# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeBase')
class nnBase(nn.Module, metaclass=ABCMeta):
    """
    Base class for GameOfLife based NNs
    Handles: save/autoload, freeze/unfreeze, casting between data formats, and training loop functions
    """
    def __init__(self):
        super().__init__()
        self.loaded    = False  # can't call sell.load() in constructor, as weights/layers have not been defined yet
        self.device    = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self.criterion = nn.BCELoss()


    def weights_init(self, layer):
        ### Default initialization seems to work best, at least for Z shaped ReLU1 - see GameOfLifeHardcodedReLU1_21.py
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            ### kaiming_normal_ corrects for mean and std of the relu function
            ### xavier_normal_ works better for ReLU6 and Z shaped activations
            if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU, nn.PReLU)):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    # small positive bias so that all nodes are initialized
                    nn.init.constant_(layer.bias, 0.1)
        else:
            # Use default initialization
            pass



    ### Prediction

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if not self.loaded: self.load()  # autoload on first function call
        return super().__call__(*args, **kwargs)



    # ### Training
    #
    # def loss(self, outputs, expected, input):
    #     return self.criterion(outputs, expected)
    #
    # def accuracy(self, outputs, expected, inputs) -> float:
    #     # noinspection PyTypeChecker
    #     return torch.sum(self.cast_int(outputs) == self.cast_int(expected)).cpu().numpy() / np.prod(outputs.shape)


    ### Freeze / Unfreeze

    def freeze(self: T) -> T:
        if not self.loaded: self.load()
        for name, parameter in self.named_parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self: T) -> T:
        if not self.loaded: self.load()
        for name, parameter in self.named_parameters():
            parameter.requires_grad = True
        return self



    ### Load / Save Functionality

    @property
    def filename(self) -> str:
        return os.path.join(os.path.dirname(__file__), 'models', f'{self.__class__.__name__}.pth')


    # DOCS: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def save(self: T, verbose=True) -> T:
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.state_dict(), self.filename)
        if verbose: print(f'{self.__class__.__name__}.savefile(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
        return self


    def load(self: T, load_weights=True, verbose=True) -> T:
        if load_weights and os.path.exists(self.filename):
            try:
                self.load_state_dict(torch.load(self.filename))
                if verbose: print(f'{self.__class__.__name__}.load(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
            except Exception as exception:
                # Ignore errors caused by model size mismatch
                if verbose: print(f'{self.__class__.__name__}.load(): model has changed dimensions, reinitializing weights\n')
                self.apply(self.weights_init)
        else:
            if verbose:
                if load_weights: print(f'{self.__class__.__name__}.load(): model file not found, reinitializing weights\n')
                # else:          print(f'{self.__class__.__name__}.load(): reinitializing weights\n')
            self.apply(self.weights_init)

        self.loaded = True    # prevent any infinite if self.loaded loops
        self.to(self.device)  # ensure all weights, either loaded or untrained are moved to GPU
        self.eval()           # default to production mode - disable dropout
        self.freeze()         # default to production mode - disable training
        return self


    ### Casting

    def cast_to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(torch.float32).to(self.device)
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(torch.float32)
            x = x.to(self.device)
            return x  # x.shape = (42,3)
        raise TypeError(f'{self.__class__.__name__}.cast_to_tensor() invalid type(x) = {type(x)}')


    ### Utility

    def print_params(self):
        print(self.__class__.__name__)
        print(self)
        for name, parameter in sorted(self.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):
            print(name)
            print(re.sub(r'\n( *\n)+', '\n', str(parameter.data.cpu().numpy())))  # remove extranious newlines
            print()
