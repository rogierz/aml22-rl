import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os
from itertools import product
from datetime import datetime


def main(input_file, show=True, output_folder=".", fname="plot"):
    df = pd.read_csv(input_file)

    fig, axs = plt.subplots(2, 2)

    fig.suptitle("average return")

    # ------------------- LR --------------------------
    x1 = df['learning_rate']
    y11 = df['source_source/avg_return']
    y12 = df['source_target/avg_return']
    y13 = df['target_target/avg_return']

    lr_values = np.array([1e-3, 2e-3, 5e-3])

    means1 = np.empty((3, len(lr_values)), dtype=float)
    stds1 = np.empty((3, len(lr_values)), dtype=float)

    y1 = [y11, y12, y13]

    for i, j in product(range(3), range(len(lr_values))):
        means1[i, j] = np.mean(y1[i][x1 == lr_values[j]])
        stds1[i, j] = np.std(y1[i][x1 == lr_values[j]])

    x1 = np.unique(x1)

    axs[0, 0].scatter(0.9 * x1, means1[0, :], label="source/source")
    axs[0, 0].scatter(x1, means1[1, :], label="source/target")
    axs[0, 0].scatter(1.1 * x1, means1[2, :], label="target/target")

    axs[0, 0].errorbar(0.9 * x1, means1[0, :], yerr=stds1[0, :], fmt='o')
    axs[0, 0].errorbar(x1, means1[1, :], yerr=stds1[1, :], fmt='o')
    axs[0, 0].errorbar(1.1 * x1, means1[2, :], yerr=stds1[2, :], fmt='o')

    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xticks(lr_values)
    axs[0, 0].set_xticklabels(lr_values)
    axs[0, 0].set_xlabel("learning rate")
    axs[0, 0].minorticks_off()

    print(axs[0, 0].get_xticklabels())

    # ------------------- gamma -----------------------

    x2 = df['gamma']
    y21 = df['source_source/avg_return']
    y22 = df['source_target/avg_return']
    y23 = df['target_target/avg_return']

    gamma_values = np.array([0.9, 0.99])

    means2 = np.empty((3, len(gamma_values)), dtype=float)
    stds2 = np.empty((3, len(gamma_values)), dtype=float)

    y2 = [y21, y22, y23]

    for i, j in product(range(3), range(len(gamma_values))):
        means2[i, j] = np.mean(y2[i][x2 == gamma_values[j]])
        stds2[i, j] = np.std(y2[i][x2 == gamma_values[j]])

    x2 = np.unique(x2)

    axs[0, 1].set_xlim([0.85, 1.04])

    axs[0, 1].scatter(x2-0.01, means2[0, :])
    axs[0, 1].scatter(x2, means2[1, :])
    axs[0, 1].scatter(x2+0.01, means2[2, :])

    axs[0, 1].errorbar(x2-0.01, means2[0, :], yerr=stds2[0, :], fmt='o')
    axs[0, 1].errorbar(x2, means2[1, :], yerr=stds2[1, :], fmt='o')
    axs[0, 1].errorbar(x2+0.01, means2[2, :], yerr=stds2[2, :], fmt='o')

    axs[0, 1].set_xticks(gamma_values)
    axs[0, 1].set_xticklabels(gamma_values)

    axs[0, 1].set_xlabel("gamma")

    # ------------------- batch -----------------------

    x3 = df['batch_size']
    y31 = df['source_source/avg_return']
    y32 = df['source_target/avg_return']
    y33 = df['target_target/avg_return']

    bs_size = np.power(2, np.arange(7, 9 + 1))

    means3 = np.empty((3, len(bs_size)), dtype=float)
    stds3 = np.empty((3, len(bs_size)), dtype=float)

    y3 = [y31, y32, y33]

    for i, j in product(range(3), range(7, 9 + 1)):
        means3[i, j - 7] = np.mean(y3[i][df.index[x3 == 2 ** j]])
        stds3[i, j - 7] = np.std(y3[i][df.index[x3 == 2 ** j]])

    x3 = np.unique(x3)

    axs[1, 0].scatter(0.9 * x3, means3[0, :])
    axs[1, 0].scatter(x3, means3[1, :])
    axs[1, 0].scatter(1.1 * x3, means3[2, :])

    axs[1, 0].errorbar(0.9 * x3, means3[0, :], yerr=stds3[0, :], fmt='o')
    axs[1, 0].errorbar(x3, means3[1, :], yerr=stds3[1, :], fmt='o')
    axs[1, 0].errorbar(1.1 * x3, means3[2, :], yerr=stds3[2, :], fmt='o')

    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].set_xticks(np.power(2, np.arange(7, 9 + 1)))
    axs[1, 0].set_xlabel("batch size")

    # ------------------- schedule ------------------------

    x4 = df['lr_schedule']

    y41 = df['source_source/avg_return']
    y42 = df['source_target/avg_return']
    y43 = df['target_target/avg_return']

    lr_schedule_values = ["constant", "linear"]

    means4 = np.empty((3, len(lr_schedule_values)), dtype=float)
    stds4 = np.empty((3, len(lr_schedule_values)), dtype=float)

    y4 = [y41, y42, y43]

    for i, j in product(range(3), range(len(lr_schedule_values))):
        means1[i, j] = np.mean(y4[i][x4 == lr_schedule_values[j]])
        stds1[i, j] = np.std(y4[i][x4 == lr_schedule_values[j]])

    x4 = pd.get_dummies(x4)['constant']
    # x4 contains the mapping: 'constant' -> 0, 'linear' -> 1
    x4 = np.unique(x4)

    interval = x4 + [-1, 1]

    axs[1, 1].set_xlim(interval)

    shift = (interval[1]-interval[0]) / 20

    axs[1, 1].scatter(x4-shift, means4[0, :])
    axs[1, 1].scatter(x4, means4[1, :])
    axs[1, 1].scatter(x4+shift, means4[2, :])

    axs[1, 1].errorbar(x4-shift, means4[0, :], yerr=stds4[0, :], fmt='o')
    axs[1, 1].errorbar(x4, means4[1, :], yerr=stds4[1, :], fmt='o')
    axs[1, 1].errorbar(x4+shift, means4[2, :], yerr=stds4[2, :], fmt='o')

    axs[1, 1].set_xticks(x4)
    axs[1, 1].set_xticklabels(lr_schedule_values)


    axs[1, 1].set_xlabel("learning rate schedule")

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
            fname += datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.savefig(fname=fname, dpi=200)


if __name__ == "__main__":
    main(os.path.join("data", "GS_data"))
