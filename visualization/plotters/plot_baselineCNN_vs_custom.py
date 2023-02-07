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
    ax.plot(x, y1, 'r', label="NatureCNN")
    ax.plot(x, y2, 'g', label="ShuffleNet V2")
    ax.plot(x, y3, 'b', label="ShuffleNet V2 (pretrained)")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
