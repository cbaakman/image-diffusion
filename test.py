#!/usr/bin/env python

import os
import random
import sys
import logging
from math import sqrt

from imageio import imread, imwrite

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from diffusion.optimizer import DiffusionModelOptimizer



_log = logging.getLogger(__name__)


def square(x: float) -> float:
    return x * x


class Model(torch.nn.Module):
    def __init__(self, T: int):
        super().__init__()

        self.T = T

        self.conv1 = torch.nn.Conv2d(4, 16, kernel_size=(32, 32), padding=16)
        self.act = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 3, kernel_size=(32, 32), padding=15)

    def forward(
        self,
        z: torch.Tensor,
        t: int,
    ):

        t_channel = (t / self.T) * torch.ones(list(z.shape[:-1]) + [1], device=z.device, dtype=torch.float)
        z = torch.cat((z, t_channel), dim=-1)

        z = z.transpose(-1, -2).transpose(-2, -3).unsqueeze(0)

        z = self.conv1(z)
        z = self.act(z)
        z = self.conv2(z)

        z = z[0].transpose(-2, -3).transpose(-1, -2)

        return z


if __name__ == "__main__":

    logging.basicConfig(filename="diffusion.log", filemode='a', level=logging.DEBUG)

    device = torch.device("cpu")

    _log.debug(f"initializing image")
    image = imread(sys.argv[1])
    image = torch.tensor((image - 127.5) / 127.5, device=device, dtype=torch.float)
    w, h, c = image.shape

    _log.debug(f"initializing model")
    T = 100
    model = Model(T).to(device=device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    nepoch = 10000
    lowest_loss = torch.tensor(1e99, device=device)
    for epoch_index in range(nepoch):
        _log.debug(f"starting epoch {epoch_index}")

        loss = dm.optimize(image)

        if loss < lowest_loss:
            torch.save(model.state_dict(), "model.pth")
            lowest_loss = loss

    model.load_state_dict(torch.load("model.pth", map_location=device))

    _log.debug(f"sampling")
    alpha = dm.alpha_function(100)
    sigma = sqrt(1.0 - square(alpha))
    epsilon = torch.randn(image.shape, device=device)
    noised = alpha * image + sigma * epsilon
    #noised = torch.randn(image.shape, device=device)
    imwrite("noised.png", (noised * 127.5 + 127.5).detach().numpy().astype("uint8"))
    image = torch.clamp(dm.sample(noised), 0.0, 1.0)

    _log.debug(f"writing image")
    imwrite("result.png", (image * 127.5 + 127.5).detach().numpy().astype("uint8"))


