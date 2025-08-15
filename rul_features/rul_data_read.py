import pandas as pd
import os
import pyarrow
import matplotlib.pyplot  as plt
import time
import numpy as np
import math
from scipy.signal import butter, filtfilt

# function for raw data revisition
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

def estimate_torque(data_read, speed_v=1800, debug=False):

    """
    Estimate the motor torque based on voltage and current inputs.
    :param v_a_raw: Voltage alpha component
    :param v_b_raw: Voltage beta component
    :param i_alpha: Current alpha component
    :param i_beta: Current beta component
    :param speed_v: Motor speed in rpm (default: 900)
    :param debug: Boolean flag to enable debugging plots (default: False)
    :return: Estimated torque array
    """

    v_a_raw = np.array(data_read["Voltage alpha downsample"])
    v_c_raw = np.array(data_read["Voltage beta downsample"])
    i_alpha = np.array(data_read["Current alpha"])
    i_beta  = np.array(data_read["Current beta"])

    # offset calibration
    v_a_raw = v_a_raw - np.mean(v_a_raw)
    v_c_raw = v_c_raw - np.mean(v_c_raw)
    i_alpha = i_alpha - np.mean(i_alpha)
    i_beta = i_beta - np.mean(i_beta)

    # Preprocess the voltage data, transform the data to alpha-beta frame
    v_alpha = v_a_raw
    v_beta = v_c_raw
    
    class EMF:
        def __init__(self):
            self.Alpha = 0.0
            self.Beta = 0.0
            self.Alpha_last = 0.0
            self.Beta_last = 0.0
            self.Alpha_LPF = 0.0
            self.Beta_LPF = 0.0
            self.Alpha_LPF_last = 0.0
            self.Beta_LPF_last = 0.0

    def emf_to_lpf(sampling_time, lpf_radius, emf_obj):
        lpf_radius_t = lpf_radius * sampling_time
        emf_coef1 = sampling_time / (lpf_radius_t + 2)
        emf_coef2 = (lpf_radius_t - 2) / (lpf_radius_t + 2)

        emf_obj.Alpha_LPF = emf_coef1 * (emf_obj.Alpha_last + emf_obj.Alpha) - emf_coef2 * emf_obj.Alpha_LPF_last
        emf_obj.Beta_LPF = emf_coef1 * (emf_obj.Beta_last + emf_obj.Beta) - emf_coef2 * emf_obj.Beta_LPF_last

        emf_obj.Alpha_last = emf_obj.Alpha
        emf_obj.Beta_last = emf_obj.Beta

        emf_obj.Alpha_LPF_last = emf_obj.Alpha_LPF
        emf_obj.Beta_LPF_last = emf_obj.Beta_LPF

    def flux_comp(omega_e, lpf_radius):
        if np.abs(omega_e) < 1:
            mag_comp = 1.0
        else:
            mag_comp = np.abs(omega_e) / np.sqrt(omega_e ** 2 + lpf_radius ** 2)

        phase_comp = -57.29578 * np.arctan2(lpf_radius, omega_e) / 360  # Degree to radians conversion

        return mag_comp, phase_comp

    # Compute necessary parameters
    fs = 20000 # Sampling rate
    flux_rs = 12.5  # Motor stator resistance
    tsim = 1 / fs  # Time step

    we = (speed_v / 60) * (np.pi * 2)*4 # electrical angular velocity
    coef = 0.2
    cross_freq = 15.0

    intgr_bw_f = max((we / (np.pi * 2)) * coef, cross_freq)
    fast_wc = intgr_bw_f * (np.pi * 2)

    # Process each time step
    emf1 = EMF()
    alpha_lpf_values, beta_lpf_values = [], []
    alpha_raw_values, beta_raw_values = [], []

    for va, vb, ia, ib in zip(v_alpha, v_beta, i_alpha, i_beta):
        emf1.Alpha = va - (ia * flux_rs)
        emf1.Beta = vb - (ib * flux_rs)
        alpha_raw_values.append(emf1.Alpha)
        beta_raw_values.append(emf1.Beta)
        emf_to_lpf(tsim, fast_wc, emf1)
        alpha_lpf_values.append(emf1.Alpha_LPF)
        beta_lpf_values.append(emf1.Beta_LPF)

    # Flux compensation
    mag_comp2, phase_comp2 = flux_comp(we, fast_wc)

    # Apply phase and magnitude compensation
    alpha_compensated_values, beta_compensated_values = [], []
    for alpha, beta in zip(alpha_lpf_values, beta_lpf_values):
        ds = alpha * mag_comp2
        qs = beta * mag_comp2
        angle = phase_comp2
        sine, cosine = np.sin(angle), np.cos(angle)
        alpha_transformed = ds * cosine - qs * sine
        beta_transformed = qs * cosine + ds * sine
        alpha_compensated_values.append(alpha_transformed)
        beta_compensated_values.append(beta_transformed)

    # Torque estimation
    torque_v = 1.5 * 4 * ((np.array(alpha_compensated_values) * i_beta) - (np.array(beta_compensated_values) * i_alpha))
    torque_avg = np.mean(np.abs(torque_v[-500:]))
    # Power and efficiency estimation
    Power_M= torque_avg*speed_v*2*np.pi/60
    Power_E = 1.5*np.mean((v_alpha * i_alpha + v_beta * i_beta))
    efficiency = Power_M / Power_E * 100
    power_sts = {
        'Power_M': Power_M,
        'Power_E': Power_E,
        'Efficiency': efficiency,
        'Efficiency_alarm': int(efficiency < 90)
    }

    if debug :
        date_time=pd.to_datetime(data_read["Unix Time"], unit='s').strftime('%Y-%m-%d %H:%M:%S')

        print("Estimated Torque:", np.mean(torque_v[-500:]))
        # print result
        for key, value in power_sts.items():
            print(f"{key}: {value}")
        time = np.arange(len(v_alpha)) * tsim

        # plot the flux values
        plt.figure(figsize=(10, 5))
        plt.plot(time, alpha_compensated_values, label='Flux Alpha')
        plt.plot(time, beta_compensated_values, label='Flux Beta ')
        plt.xlabel('Time (s)')
        plt.ylabel('Flux Values')
        plt.legend()
        plt.title('Flux Values'+date_time)
        plt.grid()
        plt.show(block=False)

        # plot the voltage values
        plt.figure(figsize=(10, 5))
        plt.plot(time, v_alpha, label='Voltage Beta (raw)')
        plt.plot(time, v_beta, label='Voltage Beta (raw)')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage Values')
        plt.legend()
        plt.title('Voltage Values')
        plt.grid()
        plt.show(block=False)

        # plot the current values
        plt.figure(figsize=(10, 5))
        plt.plot(time, i_alpha, label='Current Alpha (raw)')
        plt.plot(time, i_beta, label='Current Beta (raw)')
        plt.xlabel('Time (s)')
        plt.ylabel('Current Values')
        plt.legend()
        plt.title('Current Values')
        plt.grid()
        plt.show(block=False)

        # plot the EMF values
        plt.figure(figsize=(10, 5))
        plt.plot(time, alpha_raw_values, label='EMF Alpha (raw)')
        plt.plot(time, beta_raw_values, label='EMF Beta (raw)')
        plt.xlabel('Time (s)')
        plt.ylabel('EMF Values')
        plt.legend()
        plt.title('EMF Values')
        plt.grid()
        plt.show(block=False)

        # plot torque estimation
        plt.figure(figsize=(10, 5))
        plt.plot(time, torque_v, label='Torque (Voltage Model)')
        
        plt.plot(time[-6000:], torque_v[-6000:], label='monitored torque region', color='red')
        torque_avg=np.mean(torque_v[-6000:])
            
        plt.axhline(y=torque_avg, color='k', linestyle='--', label='Averaged torque')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N.m)')
        plt.legend()
        plt.title(f'Torque Estimation :{torque_avg:.2f} (N.m)')
        plt.grid()
        plt.show(block=False)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('EMF Alpha (Raw)', color='tab:blue')
        ax1.plot(time, alpha_raw_values, label='EMF Alpha (Raw)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('EMF Alpha (Filtered)', color='tab:red')
        # ax2.plot(time, alpha_lpf_values, label='EMF Alpha (Filtered)', color='tab:red')
        ax2.plot(time, alpha_compensated_values, label='EMF Alpha (Compensated)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')

        plt.title('EMF Alpha - Raw vs Filtered')
        plt.grid()
        plt.show(block=False)
    # return the torque value and the estimated flux
    return torque_v, alpha_compensated_values, beta_compensated_values, v_alpha, v_beta, power_sts
 
def voltage_calibrate(raw_voltage, threshold, speed=1):
   
    check_length=int (20000/(speed/3000*200)/4) # 取1/4 週期作為檢測窗長度
    calib_signal = np.copy(raw_voltage)
    
    # 從尾端往前掃描
    for i in range(len(raw_voltage) - 1, 0, -1):
        #檢查訊號差突變點
        if raw_voltage[i]- raw_voltage[i - 1] > threshold and  raw_voltage[i]*raw_voltage[i - 1]<0:
            closest_idx = None
            closest_val = float('inf') 
            for idx in range(max(i-check_length,0), i):
                if (abs(raw_voltage[idx]))<closest_val:
                    closest_val = abs(raw_voltage[idx])
                    closest_idx = idx
            # print(f"Calibrate from {max(i-check_length,0)} to {closest_idx}")
            calib_signal[closest_idx:i]= abs(raw_voltage[closest_idx:i])

    # 第二遍 從頭掃描
    for i in range(len(raw_voltage)):
        #檢查訊號差突變點
        if raw_voltage[i]- raw_voltage[i - 1] < -threshold and  raw_voltage[i]*raw_voltage[i - 1]<0:
            closest_idx = None
            closest_val = float('inf') 
            for idx in range(i, min(i+check_length,len(raw_voltage))):
                if (abs(raw_voltage[idx]))<closest_val:
                    closest_val = abs(raw_voltage[idx])
                    closest_idx = idx
            # print(f"Calibrate from {closest_idx} to {min(i+check_length,len(raw_voltage))}")
            calib_signal[i:closest_idx]= abs(raw_voltage[i:closest_idx])
            
    return calib_signal

def voltage_calibrate_stator(v_alpha, v_beta, threshold, speed=1 ):
    #轉換回 vsensing 線間電壓
    v_ab=(3*v_alpha-math.sqrt(3)*v_beta)/2
    v_bc=math.sqrt(3)*(v_beta)
    # 對 v_ab 和 v_bc 進行校準
    v_ab=voltage_calibrate(v_ab, threshold, speed)
    v_bc=voltage_calibrate(v_bc, threshold, speed)
    # v_ab=voltage_calibrate_v2(v_ab, threshold, speed)
    # v_bc=voltage_calibrate_v2(v_bc, threshold, speed)
    #轉換回 alpha beta 電壓
    v_alpha = (2*v_ab + v_bc)/3
    v_beta = v_bc/math.sqrt(3)
    return v_alpha, v_beta

def fft_integration(raw_data_alpha, raw_data_beta, Wc=10, fs=20000):
    """
    輸入複數訊號，進行FFT轉換後積分在反FFT回來
    Wc 為截止頻率
    Perform FFT integration on the provided raw data.
    :param raw_data_alpha: Raw data for alpha component
    :param raw_data_beta: Raw data for beta component
    :return: Integrated alpha and beta components
    """    
    
    # t = np.arange(0, len(raw_data_alpha))/ fs  # 時間向量
    
    # FFT
    raw_alpha_fft = np.fft.fft(raw_data_alpha)
    raw_beta_fft = np.fft.fft(raw_data_beta)
    
    freqs = np.fft.fftfreq(len(raw_data_alpha), d=1/fs)
    
    # 積分操作：除以 j2πf，排除 f=0 的成分
    raw_alpha_integrated = np.zeros_like(raw_alpha_fft)
    raw_beta_integrated = np.zeros_like(raw_beta_fft)
    
    for i, f in enumerate(freqs):
        if np.abs(f) > Wc:  # 截止頻率Hz
            # print('fft integrate')
            raw_alpha_integrated[i] = raw_alpha_fft[i] / (1j * 2 * np.pi * f)
            raw_beta_integrated[i] = raw_beta_fft[i] / (1j * 2 * np.pi * f)
        else:
            raw_alpha_integrated[i] = 0  
            raw_beta_integrated[i] = 0

    # IFFT
    raw_alpha_integrated_ifft = np.fft.ifft(raw_alpha_integrated)
    raw_beta_integrated_ifft = np.fft.ifft(raw_beta_integrated)

    # 畫圖
    # plt.figure()
    # # plt.plot(t, raw_alpha_integrated, label='integrated fft')
    # # plt.plot(freqs, raw_alpha_fft, label='Original signal')
    # plt.plot(t, raw_alpha_integrated_ifft.real, label='Integrated signal')
    # plt.xlabel('Time (s)')
    # plt.legend()
    # plt.title('Integration via FFT')
    # plt.grid(True)
    # plt.show()

    return raw_alpha_integrated_ifft.real, raw_beta_integrated_ifft.real

def calculate_thd(signal):
    """
    Calculate Total Harmonic Distortion (THD) of a 1D numpy array.
    """
    signal = np.squeeze(signal)
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N)
    fft_mags = np.abs(fft_vals)[:N // 2]
    # Find the index of the fundamental frequency (ignore DC)
    fundamental_idx = np.argmax(fft_mags[1:]) + 1
    fundamental_mag = fft_mags[fundamental_idx]
    # Harmonics: 2nd, 3rd, ..., up to Nyquist
    harmonics_mag = np.sqrt(np.sum(fft_mags[fundamental_idx*2:]**2))
    thd = harmonics_mag / fundamental_mag
    return thd


def read_rul_data(filepath, default_spd=0, default_trq=0, default_pwr=0, default_eff=0, force_recompute=False):

    Rs=12.5
    data_read = None
    # 檢查檔案是否存在
    if os.path.exists(filepath):
        if filepath.endswith('.parquet'):
            df_loaded = pd.read_parquet(filepath)
                   
                        
            data_read = {
                "Unix Time": [df_loaded["Unix Time"].iloc[0]],
                "Speed": [df_loaded["Speed"].iloc[0]], # unit rpm
                "Torque": [df_loaded["Torque"].iloc[0]],
                "Power": [df_loaded["Power"].iloc[0]],
                "Efficiency": [df_loaded["Efficiency"].iloc[0]],
                "vibration rms": [df_loaded["vibration rms"].iloc[0]] if "vibration rms" in df_loaded else [],
                # "Voltage alpha": np.array([df_loaded["Voltage alpha"].iloc[0]]).T,
                # "Voltage beta": np.array([df_loaded["Voltage beta"].iloc[0]]).T,
                "Voltage alpha": np.expand_dims(np.array([df_loaded["Voltage alpha"].iloc[0]]), axis=1).T.reshape(-1),
                "Voltage beta": np.expand_dims(np.array([df_loaded["Voltage beta"].iloc[0]]), axis=1).T.reshape(-1),
                "Current alpha": np.array([df_loaded["Current alpha"].iloc[0]]).T,  # 轉為 List
                "Current beta": np.array([df_loaded["Current beta"].iloc[0]]).T,
                "vibration data": np.array([df_loaded["raw_pico_data"].iloc[0]]).T if "raw_pico_data" in df_loaded else [],
            }
            
            if "raw_pico_data" in df_loaded:
                # 計算振動數據的均方根值
                data_read["vibration rms"] =[ np.sqrt(np.mean(np.square(np.array(data_read["vibration data"]))))]
            
            # 檢查時間是否在5/28-6/25之間 (因設電流倍率置錯誤，需要額外修正)
            unix_time = int(df_loaded["Unix Time"].iloc[0])
            start_time = 1748361600  # 2025/5/28 00:00:00
            end_time = 1750780800    # 2025/6/25 00:00:00
            
            # 電流感測器倍率錯誤修正: 如果在指定時間範圍內，電流除以10
            if start_time <= unix_time <= end_time:
                data_read["Current alpha"] = data_read["Current alpha"] / 10
                data_read["Current beta"] = data_read["Current beta"] / 10
                
            Voltage_alpha = np.array([df_loaded["Voltage alpha"].iloc[0]]).T
            Voltage_beta = np.array([df_loaded["Voltage beta"].iloc[0]]).T
            Current_alpha = data_read["Current alpha"]
            Current_beta = data_read["Current beta"]
            
                
             # 補上降採樣信號
            # down sampling from 20k to 10k  
            
            def lowpass_filter(data, cutoff=2000, fs=20000, order=2):
                data = data.flatten()
                nyq = 0.5 * fs  # 奈奎斯特頻率
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                y = filtfilt(b, a, data)
                return y
            
            try:
                current_alpha_downsample = lowpass_filter(Current_alpha, cutoff=5000, fs=20000, order=2)
                current_beta_downsample = lowpass_filter(Current_beta, cutoff=5000, fs=20000, order=2)
            except Exception as e:
                print(f"Error during downsampling: {e}")
                print("data length:", len(Current_alpha), len(Current_beta))
                print("file path:", filepath)
                raise  
            data_read["Current alpha downsample"] =current_alpha_downsample
            data_read["Current beta downsample"] = current_beta_downsample

            # 檢查是否已經有 downsample 欄位，若無則計算並存回 parquet
            need_save = False
            required_keys = [
                "Voltage alpha downsample",
                "Voltage beta downsample",
                "Current alpha downsample",
                "Current beta downsample",
                "Torque raw", 
                "Torque avg",
                "Voltage alpha thd",
                "Voltage beta thd",
                "Flux alpha",
                "Flux beta",
            ]
            
            # 確認缺少那些欄位
            missing_keys = [k for k in required_keys if k not in df_loaded.columns]
        
            if missing_keys or force_recompute:
                print("Missing keys in the parquet file, recalculating and saving:")
                # 慮波方法2 FFT暴力慮波
                # voltage_alpha_downsample, _ = filter_top_n_frequencies(Voltage_alpha, 5)
                # voltage_beta_downsample, _ = filter_top_n_frequencies(Voltage_beta, 5)
                
                # 極性重新校正
                voltage_alpha_downsample, voltage_beta_downsample = voltage_calibrate_stator(Voltage_alpha, Voltage_beta, 12, data_read["Speed"][0])
                thd_alpha = calculate_thd(voltage_alpha_downsample)
                thd_beta = calculate_thd(voltage_beta_downsample)
                voltage_alpha_downsample=lowpass_filter(voltage_alpha_downsample, cutoff=5000, fs=20000, order=2)
                voltage_beta_downsample=lowpass_filter(voltage_beta_downsample, cutoff=5000, fs=20000, order=2)
                
                # 計算反電動勢
                flux_alpha1, flux_beta1=fft_integration(voltage_alpha_downsample.squeeze()-Rs*np.array(data_read["Current alpha downsample"]), 
                                        voltage_beta_downsample.squeeze()-np.array(Rs*data_read["Current beta downsample"]), Wc=10, fs=20000)  

                data_read["Voltage alpha downsample"] = voltage_alpha_downsample
                data_read["Voltage beta downsample"] = voltage_beta_downsample
                data_read["Current alpha downsample"] = current_alpha_downsample
                data_read["Current beta downsample"] = current_beta_downsample
                data_read["Voltage alpha thd"] = thd_alpha
                data_read["Voltage beta thd"] = thd_beta
                data_read["Flux alpha"] = flux_alpha1
                data_read["Flux beta"] = flux_beta1
                

                # 計算FAST 力矩估測
                torque_raw, _, _, _, _, _=estimate_torque(data_read, speed_v=data_read["Speed"][0], debug=False)
                # 取 torque_raw 後2/3的資料取平均
                torque_avg = np.mean(torque_raw[-(len(torque_raw) * 2 // 3):])

                # 欲加入的欄位與值（確保為一維 ndarray 或 float）
                new_columns = {
                    "Voltage alpha downsample": voltage_alpha_downsample.flatten(),
                    "Voltage beta downsample": voltage_beta_downsample.flatten(),
                    "Current alpha downsample": current_alpha_downsample.flatten(),
                    "Current beta downsample": current_beta_downsample.flatten(),
                    "Torque raw": torque_raw.flatten(),
                    "Torque avg": float(torque_avg),  # 確保為 float64
                    "Voltage alpha thd": np.array([thd_alpha]),
                    "Voltage beta thd": np.array([thd_beta]),
                    "Flux alpha": flux_alpha1.flatten(),
                    "Flux beta": flux_beta1.flatten(),
                }

                # 新增欄位並確保為 object 型別（避免 numpy 寫入錯誤）
                for col, val in new_columns.items():
                    if col not in df_loaded.columns:
                        df_loaded[col] = pd.Series([None] * len(df_loaded), dtype=object)

                    # 強制轉型為一維 list（避免 pyarrow 對 ndarray 的限制）
                    if isinstance(val, np.ndarray):
                        val = val.tolist()

                    df_loaded.at[0, col] = val  # 使用 .at 避免 broadcasting 問題

                # 顯示結果確認
                # print(df_loaded.info())
                df_loaded.to_parquet(filepath, engine="pyarrow", index=False)

            # 重新載入 parquet，確保 data_read 有最新的欄位
            df_loaded = pd.read_parquet(filepath)
            data_read["Voltage alpha downsample"] = np.array(df_loaded["Voltage alpha downsample"].iloc[0])
            data_read["Voltage beta downsample"] = np.array(df_loaded["Voltage beta downsample"].iloc[0])
            # data_read["Current alpha downsample"] = np.array(df_loaded["Current alpha downsample"].iloc[0])
            # data_read["Current beta downsample"] = np.array(df_loaded["Current beta downsample"].iloc[0])
            data_read["Torque raw"] = np.array(df_loaded["Torque raw"].iloc[0])
            data_read["Torque avg"] = df_loaded["Torque avg"].iloc[0]
            data_read["Voltage alpha thd"] = df_loaded["Voltage alpha thd"].iloc[0]
            data_read["Voltage beta thd"] = df_loaded["Voltage beta thd"].iloc[0]
            data_read["Flux alpha"] = np.array(df_loaded["Flux alpha"].iloc[0])
            data_read["Flux beta"] = np.array(df_loaded["Flux beta"].iloc[0])
            # print("Data read from parquet:", data_read.keys())

        elif filepath.endswith('.csv'):
            # csv read code version
            # read time stamp from first line
            with open(filepath, "r") as file:
                first_line = file.readline().strip()  # 讀取第一行並去掉換行符
            unix_time = first_line.split(",")[1]  # 取第二個欄位 (1736773960)

            # read rest of the data
            df_loaded = pd.read_csv(filepath, skiprows=1)
            data_read = {
                "Unix Time": unix_time,
                "Speed":    default_spd,
                "Torque":   default_trq,
                "Power":    default_pwr,
                "Efficiency": default_eff,
                "Voltage alpha": df_loaded["V_alpha"].to_numpy(),
                "Voltage beta": df_loaded["V_beta"].to_numpy(),
                "Current alpha": df_loaded["I_alpha"].to_numpy(),
                "Current beta": df_loaded["I_beta"].to_numpy(),
            }
        else:
            print(f"Unsupported file format: {filepath}")
            return data_read

    else:
        print(f"檔案 {filepath} 不存在，請確認檔案路徑。")
    return data_read

def read_rul_data_v2(filepath, default_spd=0, default_trq=0, default_pwr=0, default_eff=0, force_recompute=False):
    data_read = None

    if not os.path.exists(filepath):
        print(f"檔案 {filepath} 不存在，請確認檔案路徑。")
        return None

    # 通用 FFT 過濾器 (placeholder)
    def lowpass_filter(data, cutoff=5000, fs=20000, order=2):
        data = data.flatten()
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    if filepath.endswith('.parquet'):
        df_loaded = pd.read_parquet(filepath)

        data_read = {
            "Unix Time": [df_loaded["Unix Time"].iloc[0]],
            "Speed": [df_loaded["Speed"].iloc[0]],
            "Torque": [df_loaded["Torque"].iloc[0]],
            "Power": [df_loaded["Power"].iloc[0]],
            "Efficiency": [df_loaded["Efficiency"].iloc[0]],
            "vibration rms": [df_loaded["vibration rms"].iloc[0]] if "vibration rms" in df_loaded else [],
            "Voltage alpha": np.array(df_loaded["Voltage alpha"].iloc[0]),
            "Voltage beta": np.array(df_loaded["Voltage beta"].iloc[0]),
            "Current alpha": np.array(df_loaded["Current alpha"].iloc[0]),
            "Current beta": np.array(df_loaded["Current beta"].iloc[0]),
            "vibration data": np.array(df_loaded["raw_pico_data"].iloc[0]) if "raw_pico_data" in df_loaded else [],
            # non default values
            "Voltage alpha downsample": np.array(df_loaded["Voltage alpha downsample"].iloc[0]) if "Voltage alpha downsample" in df_loaded else [],
            "Voltage beta downsample": np.array(df_loaded["Voltage beta downsample"].iloc[0]) if "Voltage beta downsample" in df_loaded else [],
            "Current alpha downsample": np.array(df_loaded["Current alpha downsample"].iloc[0]) if "Current alpha downsample" in df_loaded else [],
            "Current beta downsample": np.array(df_loaded["Current beta downsample"].iloc[0]) if "Current beta downsample" in df_loaded else [],
            
            "test" : [] if "test" not in df_loaded.columns else df_loaded["test"].iloc[0],
        }

        # RMS from vibration if missing
        if "raw_pico_data" in df_loaded and not data_read["vibration rms"]:
            data_read["vibration rms"] = [np.sqrt(np.mean(np.square(data_read["vibration data"])))]

        # 時間範圍內修正電流倍率
        unix_time = int(data_read["Unix Time"][0])
        if 1748361600 <= unix_time <= 1750780800:
            data_read["Current alpha"] /= 10
            data_read["Current beta"] /= 10

        # # 計算 current downsample
        # try:
        #     data_read["Current alpha downsample"] = lowpass_filter(data_read["Current alpha"], cutoff=5000)
        #     data_read["Current beta downsample"] = lowpass_filter(data_read["Current beta"], cutoff=5000)
        # except Exception as e:
        #     print(f"Downsampling error: {e} @ {filepath}")
        #     raise

        # # 檢查需補的欄位
        # required_keys = [
        #     "Voltage alpha downsample",
        #     "Voltage beta downsample",
        #     "Current alpha downsample",
        #     "Current beta downsample",
        #     "Torque raw", 
        #     "Torque avg"
        # ]
        # missing_keys = [k for k in required_keys if k not in df_loaded.columns]

        # if missing_keys or force_recompute:
        #     print(f"[{os.path.basename(filepath)}] 缺少欄位或強制重新計算，處理中...")

        #     # FFT 過濾處理（需自行定義）
        #     voltage_alpha_downsample, _ = filter_top_n_frequencies(data_read["Voltage alpha"], 5)
        #     voltage_beta_downsample, _ = filter_top_n_frequencies(data_read["Voltage beta"], 5)

        #     data_read["Voltage alpha downsample"] = voltage_alpha_downsample
        #     data_read["Voltage beta downsample"] = voltage_beta_downsample

        #     # 計算 torque（需自行定義）
        #     torque_raw, _, _, _, _, _ = estimate_torque(data_read, speed_v=data_read["Speed"][0], debug=False)
        #     torque_avg = np.mean(torque_raw[-(len(torque_raw) * 2 // 3):])

        #     # 儲存回 parquet
        #     new_cols = {
        #         "Voltage alpha downsample": voltage_alpha_downsample.flatten().tolist(),
        #         "Voltage beta downsample": voltage_beta_downsample.flatten().tolist(),
        #         "Current alpha downsample": data_read["Current alpha downsample"].flatten().tolist(),
        #         "Current beta downsample": data_read["Current beta downsample"].flatten().tolist(),
        #         "Torque raw": torque_raw.flatten().tolist(),
        #         "Torque avg": float(torque_avg)
        #         ""
        #     }

        #     for col, val in new_cols.items():
        #         df_loaded[col] = pd.Series([None] * len(df_loaded), dtype=object)
        #         df_loaded.at[0, col] = val

        #     df_loaded.to_parquet(filepath, engine="pyarrow", index=False)

        # # 重新讀回完整欄位
        # df_loaded = pd.read_parquet(filepath)
        # data_read["Voltage alpha downsample"] = np.array(df_loaded["Voltage alpha downsample"].iloc[0])
        # data_read["Voltage beta downsample"] = np.array(df_loaded["Voltage beta downsample"].iloc[0])
        # data_read["Torque raw"] = np.array(df_loaded["Torque raw"].iloc[0])
        # data_read["Torque avg"] = df_loaded["Torque avg"].iloc[0]

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