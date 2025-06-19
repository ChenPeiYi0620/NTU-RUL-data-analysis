"""
This function is to extract the envelope of a signal using the Hilbert transform."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rul_features')))
import numpy as np 
from scipy.signal import hilbert
from rul_data_read import read_rul_data
import matplotlib.pyplot as plt




def hilbert_envelope(signal):
    """
    Extract the envelope of a signal using the Hilbert transform.
    
    Parameters:
    signal (numpy array): The input signal.
    
    Returns:
    numpy array: The envelope of the input signal.
    """
    # Compute the analytic signal
    analytic_signal = hilbert(signal)
    
    # Compute the envelope
    envelope = np.abs(analytic_signal)
    
    return envelope



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
    
    collect_name=r"0312_V1800_10P_FREE0"
    # collect_name=r"0313_V1800_10P_FREE1"
    # collect_name=r"0314_V1800_10P_FREE2"
    
    # 整併測試資料夾路徑
    folder_path = os.path.join(NTU_dataset_path, collect_setting, collect_name)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    
    # sort file by file number 
    files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
    
    raw_rms_total=[]
    env_rms_total=[]
    
    for i, file in enumerate(files):
        test_data=read_rul_data(file)
        test_vibration_data = test_data["vibration data"]
        
        #取得包絡振福
        env=hilbert_envelope(np.array(test_vibration_data))
        # env_rms=np.sqrt(np.mean(np.square(env)))
        env_rms=np.mean(env)
        env_rms_total.append(env_rms)
        
        #取得原始振福
        raw_rms=test_data["vibration rms"][0]
        raw_rms_total.append(raw_rms)   
    
    plt.figure(figsize=(10, 5))
    plt.plot(raw_rms_total, label='Raw RMS')
    plt.plot(env_rms_total, label='Envelope RMS')
    plt.title('RMS of Raw and Envelope Signals')
    plt.xlabel('File Number')
    plt.ylabel('RMS Value')
    plt.legend()
    plt.show()  # Show plot without blocking
    #%%
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(test_vibration_data, label='Original Signal')
    plt.plot(hilbert_envelope(np.array(test_vibration_data)), label='Hilbert Envelope')
    plt.title('Hilbert Envelope')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show
    #%%
    
    
    