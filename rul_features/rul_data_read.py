import pandas as pd
import os
import pyarrow
import matplotlib.pyplot  as plt
import time
import numpy as np
from scipy.signal import butter, filtfilt

def filter_top_n_frequencies(signal, n):
    """
    保留頻譜中前 n 個最大振幅的頻率分量，其餘設為 0，並還原時域訊號。
    此函式可以避免早期不穩定vsensing訊號躁聲對訊號的影響 

    Parameters:
    signal : 1D numpy array
        輸入的時域訊號。
    n : int
        要保留的最大頻率分量個數（不含DC）。

    Returns:
    filtered_signal : 1D numpy array
        經過濾波處理後的時域訊號。
    filtered_fft : 1D numpy array (complex)
        經過濾波的頻域資訊。
    """
    signal = np.squeeze(signal)
    N = len(signal)

    # FFT 及頻率
    fft_vals = np.fft.fft(signal)
    fft_mags = np.abs(fft_vals)

    # 忽略 DC 分量（index 0）來尋找最大值
    indices = np.argsort(fft_mags[1:N//2])[-n:] + 1  # +1 是因為跳過 DC

    # 建立遮罩，只保留前 n 個
    mask = np.zeros_like(fft_vals, dtype=bool)
    mask[indices] = True
    mask[-indices] = True  # 同步保留負頻率分量

    # DC 也可以選擇保留
    mask[0] = True

    # 應用遮罩
    filtered_fft = np.zeros_like(fft_vals, dtype=complex)
    filtered_fft[mask] = fft_vals[mask]

    # IFFT 還原時域訊號
    filtered_signal = np.fft.ifft(filtered_fft).real

    return filtered_signal, filtered_fft

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
                "vibration rms": [df_loaded["vibration rms"].iloc[0]] if "vibration rms" in df_loaded else [],
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
            
            # 補上降採樣信號
            # down sampling from 20k to 10k  
            Voltage_alpha=np.array([df_loaded["Voltage alpha"].iloc[0]]).T
            Voltage_beta=np.array([df_loaded["Voltage beta"].iloc[0]]).T
            Current_alpha=np.array([df_loaded["Current alpha"].iloc[0]]).T
            Current_beta=np.array([df_loaded["Current beta"].iloc[0]]).T
            
            def lowpass_filter(data, cutoff=1000, fs=20000, order=2):
                data = data.flatten()
                nyq = 0.5 * fs  # 奈奎斯特頻率
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                y = filtfilt(b, a, data)
                return y
                
            #慮波方法1 降採樣
            # data_read["Voltage alpha downsample"] = Voltage_alpha[::2]
            # data_read["Voltage beta downsample"] = Voltage_beta[::2]  
            # data_read["Current alpha downsample"] = Current_alpha[::2]  
            # data_read["Current beta downsample"] = Current_beta[::2]  
            
            #慮波方法2 FFT暴力慮波
            data_read["Voltage alpha downsample"],_ = filter_top_n_frequencies(Voltage_alpha, 5)
            data_read["Voltage beta downsample"],_ =  filter_top_n_frequencies(Voltage_beta, 5)
            
            #慮波方法3 IIR 濾波器
            # data_read["Voltage alpha downsample"] = lowpass_filter(Voltage_alpha, cutoff=500, fs=20000, order=2)
            # data_read["Voltage beta downsample"] = lowpass_filter(Voltage_beta, cutoff=500, fs=20000, order=2) 
            data_read["Current alpha downsample"] = lowpass_filter(Current_alpha, cutoff=2000, fs=20000, order=2)
            data_read["Current beta downsample"] =lowpass_filter(Current_beta, cutoff=2000, fs=20000, order=2)
            
            
            
        
        
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