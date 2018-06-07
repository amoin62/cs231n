import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_bar(x, y, x_lablel, y_label, title, path):
    plt.figure()
    plt.xlabel(x_lablel)
    plt.ylabel(y_label)
    plt.title(title)
    plt.bar(x, y, align='center')
    plt.savefig(path)
    plt.show()


def plot_bin(x, x_lablel, title, path, bins=10):
    plt.figure()
    plt.xlabel(x_lablel)
    plt.title(title)
    plt.hist(x, bins=bins)
    plt.savefig(path)
    plt.show()