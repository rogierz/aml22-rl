"""
This file plots the data from a TPESampler search on the step 3.
"""

import os
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_utils import smooth


def main(input_file, show=True, output_folder=".", fname="plot", smoothing=0.0):
    """
    The main function takes a CSV file as input and plots the average return
    for each hyperparameter combination. The plot is saved to a .png file in the
    <output_folder>.

    :param input_file: Specify the path to the csv file containing the data
    :param show: Show the plot instead of saving it
    :param output_folder: The folder to which save the plot
    :param fname: The name of the plot file
    :param smoothing: the smoothing weight to apply to the plots, in the interval [0, 1) = [no-smoothing, max-smoothing)
    """
    df = pd.read_csv(input_file)

    df1 = df.sort_values('learning_rate')
    df2 = df.sort_values('gamma')
    df4 = df.sort_values('total_timesteps')

    fig, axs = plt.subplots(2, 2)

    fig.suptitle("average return")

    # ------------------- LR --------------------------
    x1 = df1['learning_rate']
    y11 = smooth(df1['source_source/avg_return'], smoothing)
    y12 = smooth(df1['source_target/avg_return'], smoothing)
    y13 = smooth(df1['target_target/avg_return'], smoothing)

    # axs[0, 0].scatter(df['learning_rate'], y1, s=10)
    # axs[0, 0].scatter(df['learning_rate'], y2, s=10)
    # axs[0, 0].scatter(df['learning_rate'], y3, s=10)

    # axs[0, 0].plot(df['learning_rate'], y1)
    # axs[0, 0].plot(df['learning_rate'], y2)
    # axs[0, 0].plot(df['learning_rate'], y3)

    axs[0, 0].plot(x1, y11)
    axs[0, 0].plot(x1, y12)
    axs[0, 0].plot(x1, y13)

    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xticks(np.power(10.0, np.arange(-5, -1 + 1)))
    axs[0, 0].set_xlabel("learning rate")

    # ------------------- gamma -----------------------

    x2 = df2['gamma']
    y21 = smooth(df2['source_source/avg_return'], smoothing)
    y22 = smooth(df2['source_target/avg_return'], smoothing)
    y23 = smooth(df2['target_target/avg_return'], smoothing)

    axs[0, 1].plot(x2, y21, label="source_source")
    axs[0, 1].plot(x2, y22, label="source_target")
    axs[0, 1].plot(x2, y23, label="target_target")
    axs[0, 1].set_xlabel("gamma")

    # ------------------- batch -----------------------

    # axs[1, 0].scatter(df['batch_size'], y1)
    # axs[1, 0].scatter(df['batch_size'], y2)
    # axs[1, 0].scatter(df['batch_size'], y3)

    # print(y1[df.index[df['batch_size'] == 512]])

    means = np.empty((3, 4), dtype=float)
    stds = np.empty((3, 4), dtype=float)

    y31 = df['source_source/avg_return']
    y32 = df['source_target/avg_return']
    y33 = df['target_target/avg_return']

    y = [y31, y32, y33]

    x3 = np.power(2, np.arange(7, 10 + 1))

    for i, j in product(range(3), range(7, 10 + 1)):
        means[i, j - 7] = np.mean(y[i][df.index[df['batch_size'] == 2 ** j]])
        stds[i, j - 7] = np.std(y[i][df.index[df['batch_size'] == 2 ** j]])

    axs[1, 0].scatter(x3 - .1 * x3, means[0, :])
    axs[1, 0].scatter(x3, means[1, :])
    axs[1, 0].scatter(x3 + .1 * x3, means[2, :])

    axs[1, 0].errorbar(x3 - .1 * x3, means[0, :], yerr=stds[0, :], fmt='o')
    axs[1, 0].errorbar(x3, means[1, :], yerr=stds[1, :], fmt='o')
    axs[1, 0].errorbar(x3 + .1 * x3, means[2, :], yerr=stds[2, :], fmt='o')

    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].set_xticks(np.power(2, np.arange(7, 10 + 1)))
    axs[1, 0].set_xlabel("batch size")

    # ------------------- time ------------------------

    x4 = df4['total_timesteps']

    y41 = smooth(df4['source_source/avg_return'], smoothing)
    y42 = smooth(df4['source_target/avg_return'], smoothing)
    y43 = smooth(df4['target_target/avg_return'], smoothing)

    axs[1, 1].plot(x4, y41)
    axs[1, 1].plot(x4, y42)
    axs[1, 1].plot(x4, y43)
    axs[1, 1].set_xlabel("total timesteps")
    axs[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(4, 5))

    # fig.legend(bbox_to_anchor=(0.83, 0.9))
    fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout()

    fig.subplots_adjust(top=0.85)

    if show:
        plt.show()
    else:
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        fname = os.path.join(output_folder, fname)
        if os.path.isfile(fname+".png") or os.path.basename(fname) == "plot":
            fname += datetime.now().strftime("%H-%M-%S")

        plt.savefig(fname=fname, dpi=200)


if __name__ == "__main__":
    main(os.path.join("../data", "TPE_data"))
