import pandas as pd
import matplotlib.pyplot as plt
import os 

def main():
    
    df_nature = pd.read_csv(os.path.join("..", "data", "step4_2_NATURECNN.csv"))
    df_custom = pd.read_csv(os.path.join("..", "data", "step4_2_CUSTOMNET.csv"))
    df_custom_pretrain = pd.read_csv(os.path.join("..", "data", "step4_2_CUSTOMNET_PRETRAIN.csv"))

    x = df_nature['Step']
    y1 = df_nature['Value']
    y2 = df_custom['Value']
    y3 = df_custom_pretrain['Value']

    fig,ax=plt.subplots()
    ax.plot(x, y1, 'r', label="NatureCNN (UDR)")
    ax.plot(x, y2, 'g', label="ShuffleNet V2 (UDR)")
    ax.plot(x, y3, 'b', label="ShuffleNet V2 (pretrained, UDR)")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Reward")
    ax.set_title("FrameStack model")
    
    fig.legend(ncol=2, loc='upper center')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.8)
    fig.savefig(os.path.join("..", "plots", "plot_NatureCNN_vs_customs.png"), dpi=200)

if __name__ == "__main__":
    main()
