"""
This program is to run the ccae data for the ntu platform.
"""
import tensorflow as tf
from packaging import version

# 取得目前 TensorFlow 版本
tf_version = tf.__version__
from sklearn.model_selection import train_test_split

if version.parse(tf_version) >= version.parse("2.11"):
    from keras.src.models import Model
    from keras.src.saving.saving_api import load_model
    from keras.src.layers import Input, Conv1D, Dense, Concatenate, concatenate, RepeatVector, MaxPooling1D, Activation ,UpSampling1D, Conv1DTranspose
    from keras.src.utils import plot_model
else:
    from keras.models import Model
    from keras.models import load_model
    from keras.layers import Input, Conv1D, Dense, Concatenate, concatenate, RepeatVector, MaxPooling1D, Activation ,UpSampling1D, Conv1DTranspose
    from keras.utils import plot_model
import numpy as np 
import pandas as pd
from openpyxl import Workbook
import matplotlib.pyplot as plt
import os
import sys
import csv
import time 
from IPython import embed

# include rul feature
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rul_features')))
sys.path.append(os.path.abspath("..")) 

from rul_features.rul_data_read import read_rul_data


def load_range(folder_path, start, end):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".parquet")])  # 取得所有 parquet 檔案並排序
    files = files[start-1:end]  # 取出 start 到 end 範圍內的檔案
    
    data_list = []  # 存放讀取的資料

    for file in files:
        file_path = os.path.join(folder_path, file)  # 取得完整路徑
        df = read_rul_data(file_path)  
        
        mean = np.mean(df["Current alpha"])

        # standard = 1
        standard = np.std(df["Current alpha"])
        # standard = np.max(np.abs(df["Current alpha"]))

        data_list.append((df["Current alpha"] - mean)/ standard)  # 加入清單
    
    return data_list

def data_augmentation(df, time_steps, window_size, cols=None, random_seed=None):
    # 如果未指定 cols 參數，則預設使用資料框的所有欄位
    if cols is None:
        cols = df.columns

    # 初始化一個空的列表來存放提取出的樣本數據
    samples_list = []

    # 對指定的每一列進行滑動窗口操作
    for col in cols:
        # 根據窗口大小和時間步長，從每列中提取子序列樣本
        for i in range(0, len(df) - time_steps + 1, window_size):
            # 使用 iloc 根據索引提取從 i 到 i + time_steps 的時間段的數據
            # 並將其轉換為 NumPy 陣列，方便進行後續的數據處理
            # samples_list.append(df.iloc[i:i + time_steps].to_numpy())
            samples_list.append(df[i:i + time_steps])

    # 將收集到的所有樣本轉換成 NumPy 多維陣列
    final_data = np.array(samples_list)

    # 如果指定了 random_seed，則設置隨機種子，確保數據打亂時的隨機性是可重現的
    if random_seed is not None:
        np.random.seed(random_seed)

    # 返回增強後的數據集，這是一個 NumPy 陣列
    return final_data

def Bhattacharyya_Distance(Normal_data_mse_errors, Abnormal_data_mse_errors):
    # 計算兩組數據的均值和標準差
    mu_normal, sigma_normal = np.mean(Normal_data_mse_errors), np.std(Normal_data_mse_errors)
    mu_abnormal, sigma_abnormal = np.mean(Abnormal_data_mse_errors), np.std(Abnormal_data_mse_errors)

    # 計算 Bhattacharyya 距離
    def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
        term1 = (mu1 - mu2)**2 / (sigma1**2 + sigma2**2)
        term2 = np.log((sigma1 + sigma2) / (2 * np.sqrt(sigma1 * sigma2)))
        return 0.25 * (term1 + term2)

    # 計算兩組分布的 Bhattacharyya 距離
    distance = bhattacharyya_distance(mu_normal, sigma_normal, mu_abnormal, sigma_abnormal)

    return distance



def CCAE_model(Normal_motor_data, Abnormal_motor_data, model_name, label, figure_name):

    # 預防模型名稱未打.keras
    if '.' not in model_name:
        model_name += '.keras'
    loaded_model = load_model(model_name)

    # Normal
    # 將資料做帶重疊的切片
    all_Data = []
    for i in range(len(Normal_motor_data)):
        # 將資料做帶重疊的切片
        all_Data.append(data_augmentation(Normal_motor_data[i], time_steps=1024, window_size=10, cols=[0], random_seed=42))
    Normal_final_data =  np.concatenate(all_Data)

    # 製作標籤
    labels_Normal = np.full(Normal_final_data.shape[0], label)
    # 模型預測 (samples, signal_length, num_features)
    reconstructed_Normal_data = loaded_model.predict([Normal_final_data, labels_Normal]) 
    # 將預測後之為度改為(samples, signal_length)，移除所有大小为為 1 的维度
    reconstructed_Normal_data_squeezed = np.squeeze(reconstructed_Normal_data)
    Normal_final_data = np.squeeze(Normal_final_data)  # 變成 (98,1024)
    # Calculate MSE
    Normal_data_mse_errors = np.mean(np.square(Normal_final_data - reconstructed_Normal_data_squeezed), axis=1)
    
    # Abnormal
    all_Data = []
    for i in range(len(Abnormal_motor_data)):
        # 將資料做帶重疊的切片
        all_Data.append(data_augmentation(Abnormal_motor_data[i], time_steps=1024, window_size=10, cols=[0], random_seed=42))
    Abnormal_final_data =  np.concatenate(all_Data)
    
    labels_Abnormal = np.full(Abnormal_final_data.shape[0], label)
    reconstructed_Abnormal_data = loaded_model.predict([Abnormal_final_data, labels_Abnormal])
    reconstructed_Abnormal_data_squeezed = np.squeeze(reconstructed_Abnormal_data)
    Abnormal_final_data = np.squeeze(Abnormal_final_data)  # 變成 (98,1024)
    Abnormal_data_mse_errors = np.mean(np.square(Abnormal_final_data - reconstructed_Abnormal_data_squeezed), axis=1)


    # 計算誤差相似度
    BD = Bhattacharyya_Distance(Normal_data_mse_errors, Abnormal_data_mse_errors)
    # # 將浮點數格式化為保留三位小數
    # BD = f"{BD:.3f}"
    # # 替換小數點為 'P'
    # BD = BD.replace('.', 'P')

    # # 繪製正常和異常樣本的MSE誤差分布
    # plt.figure(figsize=(10, 6))
    # plt.hist(Normal_data_mse_errors, bins=20, alpha=0.7)
    # plt.hist(Abnormal_data_mse_errors, bins=20, alpha=0.7)
    # plt.xlabel('MSE Error', fontsize=20)
    # plt.tick_params(axis='x', labelsize=20)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    # plt.ylabel('Number of Samples', fontsize=20)
    # plt.tick_params(axis='y', labelsize=20)
    # plt.savefig(figure_name+"_BD"+str(BD))
    # plt.close()

    # Save_CSV(Normal_data_mse_errors, Abnormal_data_mse_errors)

    return BD

class CCAE_model_build_train:
    
    # CCAE 模型建構 (四種模式) 
    def build_CCAE_model(input_dim_list, sequence_length=1024, condition_dim=1, model_name='test_model.keras', file_name='test_model.png'):
        inputs = []
        condition_input = Input(shape=(condition_dim,), name='condition') 
        repeated_condition = RepeatVector(sequence_length, name='repeated_condition')(condition_input)
        

        # 合併所有time-series inputs
        for idx, input_dim in enumerate(input_dim_list):
            input_layer = Input(shape=(sequence_length, input_dim), name=f'series_{idx}')
            inputs.append(input_layer)

        # 將所有輸入通道與 repeated_condition 在最後一個維度上合併
        encoder_input = Concatenate(axis=-1, name='full_encoder_input')(inputs + [repeated_condition])

        # Encoder
        x = Conv1D(filters=64, kernel_size=64, strides=16, padding='same')(encoder_input)
        x = MaxPooling1D(pool_size=2)(x)
        x = Activation('relu')(x)

        x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Activation('relu')(x)

        x = Conv1D(filters=16, kernel_size=3, padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        encoded = Activation('relu')(x)

        encoder_model = Model(inputs=inputs+[condition_input], outputs=encoded, name='Encoder')
        # encoder_model.summary()
        # plot_model(encoder_model, show_shapes=True, show_layer_names=True)



        # Decoder
        decoder_input = Input(shape=(encoder_model.output_shape[1], encoder_model.output_shape[2]), name='encoder_output')
        decoder_condition_input = Input(shape=(condition_dim,), name='decoder_condition')
        repeated_decoder_condition = RepeatVector(encoded.shape[1])(decoder_condition_input)
        decoder_merge_input = concatenate([decoder_input, repeated_decoder_condition], name='merged_decoder_input')

        # 重建tensor 
        x = Conv1DTranspose(filters=16, kernel_size=3, padding='same')(decoder_merge_input)
        x = UpSampling1D(2)(x)
        x = Activation('relu')(x)

        x = Conv1DTranspose(filters=32, kernel_size=3, padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Activation('relu')(x)

        x = Conv1DTranspose(filters=64, kernel_size=64, strides=16, padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Activation('tanh')(x)

        output_list = [Dense(dim, activation='linear', name=f'output_{idx}')(x) for idx, dim in enumerate(input_dim_list)]
        decoder_model = Model(inputs=[decoder_input, decoder_condition_input], outputs=output_list, name='Decoder')
        # decoder_model.summary()
        # plot_model(decoder_model, show_shapes=True, show_layer_names=True)

        # 編碼輸入：所有 series input + condition input
        full_encoder_inputs = inputs + [condition_input]
        encoded = encoder_model(full_encoder_inputs)

        # 將 encoder 輸出與 condition 傳入 decoder
        decoder_outputs = decoder_model([encoded, condition_input])

        # 建立完整模型
        full_model = Model(inputs=full_encoder_inputs, outputs=decoder_outputs, name=f'CCAE_model{encoder_input.shape[2]-1}_in_{len(output_list)}_out')
        
        # 儲存模型
        full_model.save(model_name)
        
        return full_model, encoder_model, decoder_model


    def CCAE_train(kearas_model, model_type:str,  model_name=[],  input_data=[], label=1,  epochs=20, batch_size=10):
        # 可直接輸入模型或根據位置載入模型
        # 如果模型輸入位為模型路徑，則直接導入
        if isinstance(kearas_model, str):
            print("Loading model from path:", kearas_model)
            model_name= kearas_model
            kearas_model = load_model(kearas_model)

        # 製作標籤
        Label = np.full(input_data.shape[0], label)
        # 先分割出80%的訓練數據和20%的驗證
        test_data_size = input_data.shape[0] * 2 // 10 # 測試數據是20%，額外算以確保各個label數量相同
        train_data, val_data, train_labels, val_labels = train_test_split(input_data, Label, test_size=test_data_size, random_state=38, shuffle=True, stratify=Label)
        
        
        # 根據訊號輸入維與類型，拆分成合適的輸入形狀 
        for i in range(input_data.shape[2]):
            if model_type == 'I2_in_I2_out':
                train_input_data=[train_data[:,:,i] for i in range(input_data.shape[2])]
                val_input_data=[val_data[:,:,i] for i in range(input_data.shape[2])]
                train_output_data=train_input_data.copy()
                val_output_data=val_input_data.copy()
            if model_type == 'V2_in_I2_out':
                train_input_data=[train_data[:,:,i] for i in range(0,2)]
                val_input_data=[val_data[:,:,i] for i in range(0,2)]
                train_output_data=[train_data[:,:,i] for i in range(2,4)]
                val_output_data=[train_data[:,:,i] for i in range(2,4)]
            if model_type == 'V2I2_in_V2I2_outs':
                train_input_data=[train_data[:,:,i] for i in range(input_data.shape[2]-1)]
                val_input_data=[val_data[:,:,i] for i in range(input_data.shape[2]-1)]
                train_output_data=train_input_data.copy()
                val_output_data=val_input_data.copy()
        
        
        kearas_model.compile(optimizer='Adam', loss='mse')
        history = kearas_model.fit(train_input_data + [train_labels.reshape(-1, 1)], train_output_data, 
                epochs= epochs,
                batch_size=batch_size,
                validation_data=(val_input_data + [val_labels.reshape(-1, 1)], val_output_data))
        kearas_model.save(model_name)
    
        return kearas_model, history
    
def CCAE_train(Motor_data,model_name):
    
    all_Data = []
    for i in range(len(Motor_data)):
        # 將資料做帶重疊的切片
        all_Data.append(data_augmentation(Motor_data[i], time_steps=1024, window_size=10, cols=[0], random_seed=42))
        if len(all_Data[i]) == 0:
            print(f"Warning: Data at index {i} is empty after augmentation.")
            continue
        print(all_Data[i].shape)
    Data =  np.concatenate(all_Data)

    # 製作標籤
    Label = np.full(Data.shape[0], 1)

    # 先分割出80%的訓練數據和20%的驗證
    test_data_size = len(Label) * 2 // 10 # 測試數據是20%，額外算以確保各個label數量相同
    train_data, val_data, train_labels, val_labels = train_test_split(
        Data, Label, test_size=test_data_size, random_state=38, shuffle=True, stratify=Label)
    
    # 時間序列和條件數據的輸入
    time_series_input = Input(shape=(1024, 1), name='series') 
    condition_input = Input(shape=(1,), name='condition')        
    condition_layer_repeated = RepeatVector(1024)(condition_input)
    merged_encoder_input = Concatenate([time_series_input, condition_layer_repeated]) 

    # encoded
    encoded_start = Conv1D(filters=64, kernel_size=64, strides=16, padding='same')(merged_encoder_input) 
    x = MaxPooling1D(pool_size=2, strides=2)(encoded_start)
    x = Activation('relu')(x)

    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    encoded = Activation('relu')(x)

    encoder_model = Model(inputs=[time_series_input, condition_input], outputs = encoded)

    # decoded
    decoder_input = Input(shape=(encoder_model.output_shape[1], encoder_model.output_shape[2]))
    decoder_condition_input_new = Input(shape=(1,), name='decoder_condition') 
    decoder_condition_input_begin = RepeatVector(encoder_model.output_shape[1])(decoder_condition_input_new)
    merged_decoder_input = concatenate([decoder_input, decoder_condition_input_begin])

    x = Conv1DTranspose(filters=16, kernel_size=3, strides=1, padding='same')(merged_decoder_input)
    x = UpSampling1D(size=2)(x)
    x = Activation('relu')(x)

    x = Conv1DTranspose(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = UpSampling1D(size=2)(x)
    x = Activation('relu')(x)

    x = Conv1DTranspose(filters=64, kernel_size=64, strides=16, padding='same')(x)
    x = UpSampling1D(size=2)(x)
    x = Activation('tanh')(x)


    decoded = Dense(1,activation='linear')(x)
    decoder_model = Model(inputs=[decoder_input, decoder_condition_input_new], outputs=decoded)

    # Full Model
    encoder_outputs = encoder_model([time_series_input, condition_input])
    decoder_outputs = decoder_model([encoder_outputs, condition_input])
 
    model = Model(inputs=[time_series_input, condition_input], outputs=decoder_outputs)
    model.compile(optimizer='Adam', loss='mse')
    # 輸出模型結構
    model.summary()
    
    def plot_model_architecture(model, file_name):     
        plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=True, rankdir='TB')

    # 做模型訓練
    history = model.fit([train_data, train_labels], train_data, 
                epochs= 20,
                batch_size=10,
                validation_data=([val_data, val_labels], val_data))
    
    # 確認檔名包含.keras
    if '.' not in model_name:
        model_name += '.keras'
    model.save(model_name)

def Save_CSV(Normal_data_mse_errors, Abnormal_data_mse_errors):
    # 將資料寫入excel
    wb = Workbook()

    # 繪製正常和異常樣本的MSE誤差分布
    ws = wb.active
    ws.title = "data1"  # 設置工作表名稱
    ws.append(['normal', '', '', '', 'abnormal', '', ''])
    ws.append(['count','left','right','','count','left','right'])

    n_normal, bins_normal, patches_normal = plt.hist(Normal_data_mse_errors, bins=20, alpha=0.7)
    n_abnormal, bins_abnormal, patches_abnormal = plt.hist(Abnormal_data_mse_errors, bins=20, alpha=0.7)
    for j in range(len(n_normal)):
        new_raw = []
        new_raw.append(n_normal[j])
        new_raw.append(bins_normal[j])
        new_raw.append(bins_normal[j+1])
        new_raw.append('')
        new_raw.append(n_abnormal[j])
        new_raw.append(bins_abnormal[j])
        new_raw.append(bins_abnormal[j+1])
        ws.append(new_raw)

    # 儲存 Excel 文件
    file_path = 'output.xlsx'
    success_flag = 0
    while success_flag == 0:
        try:
            wb.save(file_path)
            success_flag = 1
        except IOError as e:
            print(f"存檔案時發生錯誤: {e}")
            input("按Enter繼續")
    print("資料已存入"+file_path)

def plot_current(data1, data2 = np.array([])):
    
    plt.figure(figsize=(12, 8))  # Set the figure size

    plt.plot(data1, label="Current alpha")
    if data2.any():
        plt.plot(data2, label="Current alpha")
    plt.xlabel("Sample Index")
    plt.ylabel("Current")
    plt.title("Current alpha and beta")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()  # Display the figure

def write_csv(data1, data2):
    csv_output = [['RUL_2', 'RUL_5']] + list(zip(data1, data2))

    # 指定檔案名稱
    file_name = 'all_data_BD.csv'

    # 開啟檔案寫入模式
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 寫入所有數據
        writer.writerows(csv_output)
        
def get_initial_files_datalist(file_paths, initial_rfactor=0.1, input_name="Current alpha"):
    """This function is used to get the initial files for training."""
    initial_files = []
    initial_datalist=[]
    for file_path in file_paths:
        if os.path.exists(file_path):
        
            # load the files in path 
            files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(".parquet")]
            
            # sort file by file number 
            files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
            
            # Use only first 10% of files for training 
            num_files = max(int(len(files) * initial_rfactor),1)
           
            for i in range(num_files):
                file = files[i]
                # read the data
                df = read_rul_data(file)

                if len(df[input_name])<2000:
                    print('incorrect data length, please check the file: \n', file)
                    print('raw data length: ', len(df["Voltage alpha"]))
                    continue
                
                # normalization
                mean = np.mean(df[input_name])
                standard = np.std(df[input_name])
               
                initial_datalist.append((df[input_name] - mean)/ standard)  # 加入清單
                 
        else:
            print(f"File {file_path} does not exist.")
            
            
    return initial_datalist


# 執行 ccae 所需套件
class CCAE_model_application:
    def __init__(self, model=None, file_path=None):
        self.model = model
        self.file_path = file_path
        self.sort_filenames = None
        self.basic_timelist=None
        self.iput_name = "Current alpha downsample"  # 預設輸入名稱

    def predict_mse_byfile(self):
        
        # 取出檔案位置所有資料並轉換為可訓練格式
        if self.sort_filenames is None:
            files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if f.endswith(".parquet")]
            files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
            self.sort_filenames= files
            
        data_list=[]
        for i in range(len(self.sort_filenames)):
            
            # read the data
            file = self.sort_filenames[i]
            df = read_rul_data(file)
            
            # normalization
            mean = np.mean(df[self.iput_name])
            standard = np.std(df[self.iput_name])
            
            # 加入清單
            data_list.append((df[self.iput_name] - mean)/ standard)  
        
        # 將資料做帶重疊的切片    
        all_Data=[]
        time_step_in=1024
        window_size_in=10
        segment_number=int((len(data_list[0])-time_step_in)/window_size_in)+1
        for i in range(len(data_list)):
            all_Data.append(data_augmentation(data_list[i], time_steps=time_step_in, window_size=window_size_in, cols=[0], random_seed=42))
        
        data =  np.concatenate(all_Data)   
        
        labels = np.full(data.shape[0], 1)
        
        # 模型預測並記錄計算時間
        start_time=time.time()
        reconstructed_data = self.model.predict([data, labels])
        print("Predict time: ", time.time() - start_time)
        
        # 計算重建誤差
        reconstructed_data_squeezed = np.squeeze(reconstructed_data)
        data = np.squeeze(data)  # 變成 (98,1024)
        mse_errors = np.mean(np.square(data - reconstructed_data_squeezed), axis=1)
        
        # 根據 segment_number 分組並計算每組的平均值
        mse_avg_byfile = []
        mse_byfile=[]
        for i in range(len(data_list)):
            start_idx = i * segment_number
            end_idx = (i + 1) * segment_number
            mse_byfile.append(mse_errors[start_idx:end_idx])
            mse_avg_byfile.append(np.mean(mse_errors[start_idx:end_idx]))
            
        ccae_repport={
            "segment_number": segment_number,
            "mse_avg_byfile": mse_avg_byfile,
            "mse_byfile": mse_byfile,
            "mse_in_all": mse_errors,
        }
        self.ccae_repport=ccae_repport
        return ccae_repport
    def get_basic_timelist(self):
        # 取出檔案位置所有資料並轉換為可訓練格式
        if self.sort_filenames is None:
            files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if f.endswith(".parquet")]
            files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
            self.sort_filenames= files
        
        # initialize time stamp list
        data_time_list, torq_time_list, spd_time_list, pwr_time_list, eff_time_list, acc_time_list = [], [], [], [], [], [],
        first_time = None
        
        from datetime import datetime
        def parse_timestamp(timestamp):
            """將 Unix 時間轉換為 datetime 物件"""
            return datetime.fromtimestamp(timestamp)

        for file in self.sort_filenames:
            data_read = read_rul_data(file)  # 這裡假設檔案是 CSV 格式

            if first_time is None:
                first_time = data_read["Unix Time"]  # 記錄第一個檔案的起始時間
            data_time_list.append((parse_timestamp(int(data_read["Unix Time"][0])) - parse_timestamp(int(first_time[0]))).total_seconds() / 60)
            torq_time_list.append(data_read["Torque"][0])
            spd_time_list.append(data_read["Speed"][0])
            pwr_time_list.append(data_read["Power"][0])
            eff_time_list.append(data_read["Efficiency"][0])
            acc_time_list.append(data_read["vibration rms"][0] if data_read["vibration rms"] else None)

        basic_extract_result={
            "Time stamps":data_time_list,
            "torque_time_list":torq_time_list,
            "speed_time_list": spd_time_list,
            "power_time_list": pwr_time_list,
            "efficiency_time_list": eff_time_list,
            "vibration_time_list": acc_time_list,
        }
        self.basic_timelist=basic_extract_result
        return basic_extract_result

if __name__ == "__main__":

    # This program is used to test the CCAE model for 1800 rpm data  
    train_file_list=[]
    
    # Data set folder 
    NTU_dataset_path = r'D:\OneDrive\RUL HI Reasearch Result\Data_sets\NTU_RUL_v2_data\Acc_life_test_data\Organized_Data'
    
    # 測試階段只引入 1800rpm的資料
    collect_setting = r"Load_Free\Speed_1800\Pressure_10psi"  
    train_file_list.append(os.path.join(NTU_dataset_path, collect_setting, r"0312_V1800_10P_FREE0"))
    train_file_list.append(os.path.join(NTU_dataset_path, collect_setting, r"0313_V1800_10P_FREE1"))
    train_file_list.append(os.path.join(NTU_dataset_path, collect_setting, r"0314_V1800_10P_FREE2"))

    collect_setting = r"Load_1V\Speed_1800\Pressure_10psi"
    train_file_list.append(os.path.join(NTU_dataset_path, collect_setting, r"10kg_1V_1800rpm_1"))
    train_file_list.append(os.path.join(NTU_dataset_path, collect_setting, r"15kg_1V_1800rpm_1"))
    train_file_list.append(os.path.join(NTU_dataset_path, collect_setting, r"15kg_1V_1800rpm_2"))
    
    collect_setting = r"Load_1V\Speed_1800\Pressure_5psi"
    # train_file_list.appeng(os.path.join(NTU_dataset_path, collect_setting, r"5psi_1V_1_3min"))
    train_file_list.append(os.path.join(NTU_dataset_path, collect_setting, r"06kg_1V_1800rpm_2"))

    Is_predict=1
    # Is_predict=0
    
    if Is_predict==0:
    
        # use the first 10% data for CCAE training or filtering 
        Pre_train_set=0.1
        # Create training data set 
        motor_data_list=get_initial_files_datalist(train_file_list, Pre_train_set,
                                                   input_name="Current alpha downsample")
      
        
        # Shuffle the training data randomly
        np.random.seed(42)  # Set random seed for reproducibility 
        np.random.shuffle(motor_data_list)

        # 訓練CCAE 模型
        
        CCAE_train(Motor_data = motor_data_list,
                                model_name = "0524_ItoI_model_downsample.keras")
        
    # 簡單套模檢視
    if Is_predict == 1 : 
        # 指定模型名稱
        # model_name = "0523_test_model.keras"
        model_name = "0524_ItoI_model_downsample.keras"
        
        
        ccae_model=load_model(model_name)
        
        # 建立 ccae 應用物件
        my_ccaes=[]
        for i in range(len(train_file_list)):
            my_ccae=CCAE_model_application(model=ccae_model, file_path=train_file_list[i])
            my_ccaes.append(my_ccae)
        mse_errors = my_ccae.predict_mse_byfile()
        my_ccae.get_basic_timelist()
        
        plt.figure(figsize=(12, 8))
        plt.plot(my_ccae.basic_timelist["Time stamps"], mse_errors["mse_avg_byfile"], label="MSE Errors")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE Error")
        plt.title("MSE Errors of CCAE Model")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)
        
        plt.savefig("mse_errors_plot.png")
       
       #%% 
    
 
