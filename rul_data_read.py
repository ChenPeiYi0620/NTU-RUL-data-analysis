import pandas as pd
import os
import pyarrow
import matplotlib.pyplot  as plt
import time
import numpy as np

def read_rul_data(filepath, default_spd=0, default_trq=0, default_pwr=0, default_eff=0):

    data_read = None
    # 檢查檔案是否存在
    if os.path.exists(filepath):
        if filepath.endswith('.parquet'):
            df_loaded = pd.read_parquet(filepath)
            data_read = {
                "Unix Time": [df_loaded["Unix Time"].iloc[0]],
                "Speed": [df_loaded["Speed"].iloc[0]],
                "Torque": [df_loaded["Torque"].iloc[0]],
                "Power": [df_loaded["Power"].iloc[0]],
                "Efficiency": [df_loaded["Efficiency"].iloc[0]],
                "vibration rms": [df_loaded["vibration rms"].iloc[0]] if "vibration rms" in df_loaded else 0,
                "Voltage alpha": np.array([df_loaded["Voltage alpha"].iloc[0]]).T,
                "Voltage beta": np.array([df_loaded["Voltage beta"].iloc[0]]).T,
                "Current alpha": np.array([df_loaded["Current alpha"].iloc[0]]).T,  # 轉為 List
                "Current beta": np.array([df_loaded["Current beta"].iloc[0]]).T,
                "vibration data": np.array([df_loaded["raw_pico_data"].iloc[0]]).T if "raw_pico_data" in df_loaded else [],
            }
            if "raw_pico_data" in df_loaded:
                # 計算振動數據的均方根值
                # 這裡假設 raw_pico_data 是一個一維數組
                data_read["vibration rms"] =[ np.sqrt(np.mean(np.square(np.array(data_read["vibration data"]))))]
        elif filepath.endswith('.csv'):
            # csv read code version
            # read time stamp from first line
            with open(filepath, "r") as file:
                first_line = file.readline().strip()  # 讀取第一行並去掉換行符
            unix_time = first_line.split(",")[1]  # 取第二個欄位 (1736773960)

            # read rest of the data
            df_loaded=pd.read_csv(filepath, skiprows=1)
            data_read = {
                "Unix Time": unix_time,
                "Speed":    default_spd,
                "Torque":   default_trq,
                "Power":    default_pwr,
                "Efficiency": default_eff,
                "Voltage alpha": df_loaded["V_alpha"].to_numpy(),
                "Voltage beta": df_loaded["V_beta"].to_numpy(),
                "Current alpha":df_loaded["I_alpha"].to_numpy(),
                "Current beta": df_loaded["I_beta"].to_numpy(),
            }
        else:
            print(f"Unsupported file format: {filepath}")
            return data_read


    else:
        print(f"檔案 {filepath} 不存在，請確認檔案路徑。")
    return data_read

def sort_rul_files(rul_file_path)

if __name__ == '__main__':
    # plot the read data
    # 指定 Parquet 檔案名稱
    parquet_file = "RUL_v2_record/06kg_1V_1800rpm_1/RUL_Data_3_2.parquet"
    data_read = read_rul_data(parquet_file)

    plt.figure(figsize=(12, 8))  # Set the figure size

    # First subplot for Voltage alpha and Voltage beta
    plt.subplot(2, 1, 1)
    plt.plot(data_read["Voltage alpha"], label="Voltage alpha")
    plt.plot(data_read["Voltage beta"], label="Voltage beta")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage")
    plt.title("Voltage alpha and beta")
    plt.legend()
    plt.grid(True)

    # Second subplot for Current alpha and Current beta
    plt.subplot(2, 1, 2)
    plt.plot(data_read["Current alpha"], label="Current alpha")
    plt.plot(data_read["Current beta"], label="Current beta")
    plt.xlabel("Sample Index")
    plt.ylabel("Current")
    plt.title("Current alpha and beta")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()  # Display the figure