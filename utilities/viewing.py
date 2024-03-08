from typing import Tuple

import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.ticker as tick
from torch import Tensor

from data.model import SpeciesConfiguration


def plot_spectrogram(
                     spectrogram: Tensor,
                     configuration: SpeciesConfiguration,
                     log=False,
                     axes=None,
                     **kwargs
                     ):
    kwargs.setdefault("cmap", plt.cm.get_cmap("viridis"))
    kwargs.setdefault("rasterized", True)
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.squeeze().cpu().numpy()
    spectrogram = spectrogram.T
    figsize: Tuple[int, int] = (5, 10)
    figure = plt.figure(figsize=figsize)
    if log:
        f = np.logspace(np.log2(configuration.minimum_frequency), np.log2(configuration.maximum_frequency), num=spectrogram.shape[0], base=2)
    else:
        f = np.linspace(configuration.minimum_frequency, configuration.maximum_frequency, num=spectrogram.shape[0])
    t = np.arange(0, spectrogram.shape[1]) * configuration.hop_length / configuration.hop_length
    if axes is None:
        axes = plt.gca()

    axes.pcolormesh(t, f, spectrogram, shading="auto", **kwargs)

    axes.set_xlim(t[0], t[-1])
    axes.set_ylim(f[0], f[-1])
    if log:
        axes.set_yscale("symlog", basey=2)
    yaxis = axes.yaxis
    yaxis.set_major_formatter(tick.ScalarFormatter())
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")

    plt.show()