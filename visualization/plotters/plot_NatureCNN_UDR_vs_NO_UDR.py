import pandas as pd
import matplotlib.pyplot as plt
import os 

def main():
    df_step4 = pd.read_csv(os.path.join("..", "data", "step4_noUDR.csv"))
    df_step4_UDR = pd.read_csv(os.path.join("..", "data", "step4_UDR.csv"))
    
    x = df_step4['Step']
    y1 = df_step4['Value']
    y2 = df_step4_UDR['Value']

    fig,ax=plt.subplots(figsize=(7,6))
    ax.plot(x, y1, 'r', label="NatureCNN (No UDR)")
    ax.plot(x, y2, 'r--', label="NatureCNN (UDR)")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Reward")
    ax.set_title("NatureCNN: UDR vs No UDR")

    fig.legend(ncol=2, loc="upper center")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9)
    fig.savefig(os.path.join("..", "plots", "plot_NatureCNN_UDR_vs_NO_UDR.png"), dpi=200)


if __name__ == "__main__":
    main()