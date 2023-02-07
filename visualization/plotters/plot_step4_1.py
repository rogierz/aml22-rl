import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    plt.rcParams.update({'font.size': 8})
<<<<<<< HEAD
    title_list = ["Masses of target \n reduced by 50% (except torso)",
                  "Masses of target \n reduced by 25% (except torso)", "Masses of target not modified",
                  "Masses of target \n increased by 25%", "Masses of target \n increased by 50%"]
    for i in range(0, 4):
        df_step4_1_maximizeCNN = pd.read_csv(
            os.path.join("..", "data", f"step4_1", f"step4_1_{i}_test_MAXIMIZE_CNN_log.csv"))
        df_step4_1_maximizeMLP = pd.read_csv(
            os.path.join("..", "data", "step4_1", f"step4_1_{i}_test_MAXIMIZE_MLP_log.csv"))
        df_step4_1_minimizeCNN = pd.read_csv(
            os.path.join("..", "data", "step4_1", f"step4_1_{i}_test_MINIMIZE_CNN_log.csv"))
        df_step4_1_minimizeMLP = pd.read_csv(
            os.path.join("..", "data", "step4_1", f"step4_1_{i}_test_MINIMIZE_MLP_log.csv"))

        fig, ax = plt.subplots(figsize=(7, 6))
=======
    title_list = ["Masses of target \n reduced by 50% (except torso)", "Masses of target \n reduced by 25% (except torso)", "Masses of target \n not modified", "Masses of target \n increased by 25% (except torso)", "Masses of target \n increased by 50% (except torso)"]
    for i in range(0,5):
        df_step4_1_maximizeCNN = pd.read_csv(os.path.join("..", "data", f"step4_1", f"step4_1_{i}_test_MAXIMIZE_CNN_log.csv"))
        df_step4_1_maximizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", f"step4_1_{i}_test_MAXIMIZE_MLP_log.csv"))
        df_step4_1_minimizeCNN = pd.read_csv(os.path.join("..", "data", "step4_1", f"step4_1_{i}_test_MINIMIZE_CNN_log.csv"))
        df_step4_1_minimizeMLP = pd.read_csv(os.path.join("..", "data", "step4_1", f"step4_1_{i}_test_MINIMIZE_MLP_log.csv"))
>>>>>>> 5ab4992a (updated plot step 4.1)

        x = df_step4_1_maximizeCNN['Step']
        y1 = df_step4_1_maximizeCNN['Value']
        y2 = df_step4_1_maximizeMLP['Value']
        y3 = df_step4_1_minimizeCNN['Value']
        y4 = df_step4_1_minimizeMLP['Value']
        ax.set_box_aspect(1)
        ax.plot(x, y1, 'r', label="Nature CNN (Maximize)", linewidth=3)
        ax.plot(x, y2, 'g', label="MLP (Maximize)", linewidth=3)
        ax.plot(x, y3, 'r--', label="Nature CNN (Minimize)", linewidth=3)
        ax.plot(x, y4, 'g--', label="MLP (Minimize)", linewidth=3)
        ax.set_xlabel("Episode number")
        ax.set_ylabel("Reward")
        ax.set_title(title_list[i])

        fig.savefig(os.path.join("..", "plots", f"plot_step4_1_{i}.png"), dpi=200)


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
