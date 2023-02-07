import pandas as pd
import matplotlib.pyplot as plt
import os 

def main():
    df_step4 = pd.read_csv(os.path.join("..", "data", "step4_NO_UDR.csv"))
    #NOTE: load data from trial with the same hyperparameters as step 4
    df_step2_3 = pd.read_csv(os.path.join("..", "data", "step2_3.csv"))
    


    x = df_step4['Step']
    y1 = df_step4['Value']
    y2 = df_step2_3['Value']

    fig,ax=plt.subplots()
    ax.plot(x, y1, 'r', label="NatureCNN (No UDR)")
    ax.plot(x, y2, 'g', label="SAC (No UDR)")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()