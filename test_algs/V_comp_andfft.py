"""
This function is to help the fft and denoising the v-sensing voltage."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rul_features')))
import numpy as np 
from scipy.signal import hilbert
from rul_data_read import read_rul_data
import matplotlib.pyplot as plt





def real_fft(signal, sampling_rate):
    """
    Perform FFT on the signal and return the frequency and amplitude.
    
    Parameters:
    signal (numpy array): The input signal.
    fs (float): Sampling frequency.
    
    Returns:
    tuple: Frequency and amplitude of the FFT.
    """
    signal = signal.flatten()
    n = len(signal)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/sampling_rate))
    fft_result = np.fft.fft(signal, n=n)
    fft_vals_shifted = np.fft.fftshift(fft_result)/ n
    fft_vals_shifted_db = 20 * np.log10(np.abs(fft_vals_shifted) )
    return freqs, fft_vals_shifted, fft_vals_shifted_db

def complex_fft(signal_real: np.ndarray, signal_imag: np.ndarray, sampling_rate: float):
    signal_complex = signal_real + 1j * signal_imag
    signal_complex = signal_complex.flatten()
    N = len(signal_complex)
    fft_vals = np.fft.fft(signal_complex, n=N)
    fft_vals_shifted = np.fft.fftshift(fft_vals)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sampling_rate))
    # fft_result = np.abs(fft_vals_shifted) / N
    fft_result=fft_vals_shifted / N
    return freqs, fft_result

def voltage_denoise_vs(raw_voltage :np.ndarray, threshold=5):
    
    """
    Denoise the v-sensing voltage signal using moving filter.
    
    Parameters:
    raw_voltage (numpy array): The input voltage signal.
    
    Returns:
    numpy array: The denoised voltage signal.
    """
    # Make a copy of raw_voltage to avoid modifying the original array
    dn_voltage = raw_voltage.copy()
    
    i = 3
    while i < len(raw_voltage) - 2:
        if np.abs(raw_voltage[i] - raw_voltage[i - 1]) > threshold:
            dn_voltage[i] = raw_voltage[i - 1]
            dn_voltage[i + 1] = raw_voltage[i + 2]
            i += 3  # 跳過下一個點
        else:
            i += 1  # 正常前進
            
    return dn_voltage


if __name__ == "__main__":
   
    try:
        get_ipython().run_line_magic('matplotlib', 'widget')
    except:
        pass

    #%%
    # 測試資料夾位置
    NTU_dataset_path = r'D:\OneDrive\RUL HI Reasearch Result\Data_sets\NTU_RUL_v2_data\Acc_life_test_data\Organized_Data'
    
    # test_datapath = r"Load_Free\Speed_1800\Pressure_10psi\0314_V1800_10P_FREE2"
    collect_setting = r"Load_Free\Speed_1800\Pressure_10psi"
    
    # collect_name=r"0312_V1800_10P_FREE0"
    # collect_name=r"0313_V1800_10P_FREE1"
    collect_name=r"0314_V1800_10P_FREE2"
    
    # 整併測試資料夾路徑
    folder_path = os.path.join(NTU_dataset_path, collect_setting, collect_name)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    
    # sort file by file number 
    files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
    
    test_data=read_rul_data(files[329])
    v_alpha=np.array(test_data["Voltage alpha"])
    v_beta=np.array(test_data["Voltage beta"])
    
    v_alpha_dn=voltage_denoise_vs(v_alpha)
    v_beta_dn=voltage_denoise_vs(v_beta)
 
    freqs, valpha_fft, valpha_fft_db=real_fft(v_alpha, sampling_rate=20000) 
    freqs, valpha_dn_fft, valpha_dn_fft_db=real_fft(v_alpha_dn, sampling_rate=20000)
    
    #sigle signal slice plot 
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(v_alpha_dn, label='Voltage Alpha')
    plt.plot(v_beta_dn, label='Denoise Voltage Alpha')
    plt.title('Voltage Alpha')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(freqs, valpha_fft_db, label='FFT')
    plt.plot(freqs, valpha_dn_fft_db, label='Denoise FFT')
    plt.title('FFT of Voltage Alpha')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()  
    
    
    #%%fft slice with time  
    fft_list=[]
    for i, file in enumerate(files):
        test_data=read_rul_data(file)
        i_alpha=np.array(test_data["Current alpha"])
        i_beta=np.array(test_data["Current beta"])
        
        freqs, valpha_fft=complex_fft(i_alpha, i_beta, sampling_rate=20000) 
        fft_info=np.vstack((freqs, 20*np.log10(np.abs(valpha_fft))))
        
        fft_list.append(fft_info)
        
    # 限制取圖範圍
    f_limit = 500
    freq_mask = np.abs(freqs) <= f_limit
    freqs = freqs[freq_mask]
    
    # 取出每個檔案的 FFT 結果
    num_segments = len(fft_list)
    num_freqs = fft_list[0].shape[1]

    # 建立 amplitude 矩陣 Z：shape (freqs, segments)
    Z = np.array([seg[1, :] for seg in fft_list]).T  # shape: (4000, N)
    Z = Z[freq_mask, :]
    
    # 繪製 2D heatmap
    plt.figure(figsize=(12, 8))

    extent = [0, num_segments, freqs[0], freqs[-1]]  # [Xmin, Xmax, Ymin, Ymax]

    plt.imshow(Z, aspect='auto', origin='lower', extent=extent, cmap='viridis', vmin=-60, vmax=20)
    plt.colorbar(label='Amplitude')

    plt.xlabel('Index (Time Segment)')
    plt.ylabel('Frequency (Hz)')
    plt.title('2D FFT Heatmap (Amplitude vs Frequency vs Time)')
    plt.tight_layout()
    plt.show()
   
    #%%
    
    
    