
import pandas as pd
import matplotlib.pyplot as plt
import os 

def main():

    df_step2_3 = pd.read_csv(os.path.join("..", "data", "step2_3_trial0.csv"))
    x = df_step2_3['Step']
    y1 = df_step2_3['Value']

    df_step3 = pd.read_csv(os.path.join("..", "data", "step3_trial6.csv"))
    y2 = df_step3['Value']

    df_step4 = pd.read_csv(os.path.join("..", "data", "step4_noUDR.csv"))
    y3 = df_step4['Value']

    fig,ax=plt.subplots()
    ax.plot(x, y1, 'r', label="SAC MLP (No UDR)")
    ax.plot(x, y2, 'g', label="SAC MLP (UDR)")
    ax.plot(x, y3, 'b', label="SAC CNN (NO UDR)")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Reward")
    ax.set_title("Baseline vs UDR model")

    fig.legend(ncol=2, loc='upper center')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.savefig(os.path.join("..", "plots", "plot_step4_vs_step2_3.png"), dpi=200)

if __name__ == "__main__":
    main()
