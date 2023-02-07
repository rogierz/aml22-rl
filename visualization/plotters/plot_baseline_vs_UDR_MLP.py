
import pandas as pd
import matplotlib.pyplot as plt
import os 

def main():

    df_step2_3 = pd.read_csv(os.path.join("..", "data", "step_2_3_best_trial_16_source_target.csv"))
    x = df_step2_3['Step']
    y1 = df_step2_3['Value']

    df_step3 = pd.read_csv(os.path.join("..", "data", "step3_best_params_v1_sac_tb_step3v1_log_trial_6.csv"))
    y2 = df_step3['Value']

    df_step4 = pd.read_csv(os.path.join("..", "data", "step4_noUDR.csv"))
    y3 = df_step4['Value']


    fig,ax=plt.subplots()
    ax.plot(x, y1, 'r')
    ax.plot(x, y2, 'g')
    ax.plot(x, y3, 'b')
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Reward")
    plt.show()

if __name__ == "__main__":
    main()
