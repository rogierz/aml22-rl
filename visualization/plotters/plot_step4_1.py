import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    plt.rcParams.update({'font.size': 8})

    df_step4_1_maximizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_0_test_MAXIMIZE_CNN_log.csv"))
    df_step4_1_maximizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_0_test_MAXIMIZE_MLP_log.csv"))
    df_step4_1_minimizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_0_test_MINIMIZE_CNN_log.csv"))
    df_step4_1_minimizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_0_test_MINIMIZE_MLP_log.csv"))

    fig,ax = plt.subplots()
    
    x = df_step4_1_maximizeCNN['Step']
    y1 = df_step4_1_maximizeCNN['Value']
    y2 = df_step4_1_maximizeMLP['Value']
    y3 = df_step4_1_minimizeCNN['Value']
    y4 = df_step4_1_minimizeMLP['Value']
    ax.set_box_aspect(1)
    ax.plot(x, y1, 'r', label="Nature CNN (Maximize)")
    ax.plot(x, y2, 'g', label="MLP (Maximize)")
    ax.plot(x, y3, 'b', label="Nature CNN (Minimize)")
    ax.plot(x, y4, 'y', label="MLP (Minimize)")
    ax.set_xlabel("Episode number", fontsize = 8)
    ax.set_ylabel("Reward",  fontsize = 8)
    ax.set_title("Masses of target \n reduced by 50% (except torso)", fontsize=8)

    fig.savefig(os.path.join("..", "plots", "plot_step4_1_0.png"), dpi=200)
    return
    fig,ax = plt.subplots()
    df_step4_1_maximizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_1_test_MAXIMIZE_CNN_log.csv"))
    df_step4_1_maximizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_1_test_MAXIMIZE_MLP_log.csv"))
    df_step4_1_minimizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_1_test_MINIMIZE_CNN_log.csv"))
    df_step4_1_minimizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_1_test_MINIMIZE_MLP_log.csv"))
    y1 = df_step4_1_maximizeCNN['Value']
    y2 = df_step4_1_maximizeMLP['Value']
    y3 = df_step4_1_minimizeCNN['Value']
    y4 = df_step4_1_minimizeMLP['Value']
    ax[1].set_box_aspect(1)
    ax[1].plot(x, y1, 'r')
    ax[1].plot(x, y2, 'g')
    ax[1].plot(x, y3, 'b')
    ax[1].plot(x, y4, 'y')
    ax[1].set_xlabel("Episode number", fontsize=8)
    # ax[1].set_ylabel("Reward", fontsize=8)
    # ax[1].set_title("Masses of target \n reduced by 25% (except torso)", fontsize=8)

    df_step4_1_maximizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_2_test_MAXIMIZE_CNN_log.csv"))
    df_step4_1_maximizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_2_test_MAXIMIZE_MLP_log.csv"))
    df_step4_1_minimizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_2_test_MINIMIZE_CNN_log.csv"))
    df_step4_1_minimizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_2_test_MINIMIZE_MLP_log.csv"))
    y1 = df_step4_1_maximizeCNN['Value']
    y2 = df_step4_1_maximizeMLP['Value']
    y3 = df_step4_1_minimizeCNN['Value']
    y4 = df_step4_1_minimizeMLP['Value']
    ax[2].set_box_aspect(1)
    ax[2].plot(x, y1, 'r')
    ax[2].plot(x, y2, 'g')
    ax[2].plot(x, y3, 'b')
    ax[2].plot(x, y4, 'y')
    ax[2].set_xlabel("Episode number", fontsize = 8)
    ax[2].set_ylabel("Reward", fontsize = 8)
    ax[2].set_title("Masses of target not modified", fontsize=8)

    df_step4_1_maximizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_3_test_MAXIMIZE_CNN_log.csv"))
    df_step4_1_maximizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_3_test_MAXIMIZE_MLP_log.csv"))
    df_step4_1_minimizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_3_test_MINIMIZE_CNN_log.csv"))
    df_step4_1_minimizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_3_test_MINIMIZE_MLP_log.csv"))
    y1 = df_step4_1_maximizeCNN['Value']
    y2 = df_step4_1_maximizeMLP['Value']
    y3 = df_step4_1_minimizeCNN['Value']
    y4 = df_step4_1_minimizeMLP['Value']
    ax[3].set_box_aspect(1)
    ax[3].plot(x, y1, 'r')
    ax[3].plot(x, y2, 'g')
    ax[3].plot(x, y3, 'b')
    ax[3].plot(x, y4, 'y')
    ax[3].set_xlabel("Episode number", fontsize=8)
    # ax[3].set_ylabel("Reward", fontsize=8)
    # ax[3].set_title("Masses of target \n increased by 25% (except torso)", fontsize=8)

    df_step4_1_maximizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_4_test_MAXIMIZE_CNN_log.csv"))
    df_step4_1_maximizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_4_test_MAXIMIZE_MLP_log.csv"))
    df_step4_1_minimizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_4_test_MINIMIZE_CNN_log.csv"))
    df_step4_1_minimizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", "step4_1_4_test_MINIMIZE_MLP_log.csv"))
    y1 = df_step4_1_maximizeCNN['Value']
    y2 = df_step4_1_maximizeMLP['Value']
    y3 = df_step4_1_minimizeCNN['Value']
    y4 = df_step4_1_minimizeMLP['Value']
    ax[4].set_box_aspect(1)
    ax[4].plot(x, y1, 'r')
    ax[4].plot(x, y2, 'g')
    ax[4].plot(x, y3, 'b')
    ax[4].plot(x, y4, 'y')
    ax[4].set_xlabel("Episode number", fontsize=8)
    # ax[4].set_ylabel("Reward", fontsize=8)
    # ax[4].set_title("Masses of target \n increased by 50% (except torso)", fontsize=8)

    fig.legend(ncol=4, loc='upper center')
    fig.set_size_inches(10, 2)
    # plt.subplots_adjust(wspace=0.5, hspace=0.1, left=0.07, bottom=0.13)
    fig.tight_layout()
    plt.subplots_adjust(top=0.8)
    fig.savefig(os.path.join("..", "plots", "plot_step4_1.png"), dpi=200)


if __name__ == "__main__":
    plt.rcParams.update({
        # "text.usetex": True,
        "font.family": "sans-serif",
        # "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        # "legend.fontsize": 18,
        "figure.dpi": 180
    })
    main()
