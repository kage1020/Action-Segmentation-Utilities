import pathlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def plot_feature(feature, file_path: str):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        axfig = ax.imshow(feature, aspect='auto', interpolation='none', cmap='jet')
        fig.colorbar(axfig, ax=ax)
        fig.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)
        plt.savefig(file_path)
