#!/usr/bin/env python

import os
import random
import sys
import logging

from imageio import imread, imwrite

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from diffusion.optimizer import DiffusionModelOptimizer



_log = logging.getLogger(__name__)


class Model(torch.nn.Module):
    def __init__(self, T: int, w: int, h: int, c: int):
        super(Model, self).__init__()

        self.T = T
        self.shape = (w, h, c)

        trans = 128

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(w * h * c, trans),
            torch.nn.ReLU(),
            torch.nn.Linear(trans, w * h * c),
        )

    def forward(
        self,
        z: torch.Tensor,
        t: int,
    ) -> torch.Tensor:

        shape = z.shape
        z = z.reshape(-1)

        e = self.mlp(z)

        return e.reshape(shape)


if __name__ == "__main__":

    logging.basicConfig(filename="diffusion.log", filemode='a', level=logging.DEBUG)

    device = torch.device("cpu")

    _log.debug(f"initializing image")
    image = imread(sys.argv[1])
    image = torch.tensor(image / 255.0, device=device, dtype=torch.float)
    w, h, c = image.shape
    image.unsqueeze(0)

    _log.debug(f"initializing model")
    T = 10
    model = Model(T, w, h, c).to(device=device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    nepoch = 100
    for epoch_index in range(nepoch):
        _log.debug(f"starting epoch {epoch_index}")

        dm.optimize(image)

        torch.save(model.state_dict(), "model.pth")

    model.load_state_dict(torch.load("model.pth", map_location=device))

    _log.debug(f"sampling")
    image = dm.sample() * 255

    _log.debug(f"writing image")
    imwrite("result.png", image.detach().numpy().astype("uint8"))


