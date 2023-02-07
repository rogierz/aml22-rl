import pandas as pd
import matplotlib.pyplot as plt
import os 

def main():
    df_step4_UDR = pd.read_csv(os.path.join("..", "data", "step4_UDR.csv"))
    df_step4_NO_UDR = pd.read_csv(os.path.join("..", "data", "step4_NOUDR.csv"))
    df_step2_3 = pd.read_csv(os.path.join("..", "data", "step2_3.csv"))


    x = df_step4_UDR['Step']
    y1 = df_step4_UDR['Value']
    y2 = df_step4_NO_UDR['Value']
    y3 = df_step2_3['Value']

    fig,ax=plt.subplots()
    ax.plot(x, y1, 'r', label="Nature CNN (UDR)")
    ax.plot(x, y2, 'g', label="Nature CNN (NO UDR)")
    ax.plot(x, y3, 'b', label="SAC (NO UDR)")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()