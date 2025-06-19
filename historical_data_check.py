import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import rul_data_read as rul_rd
import time

import matplotlib.animation as animation
from IPython.display import HTML, display, IFrame

def parse_timestamp(timestamp):
    """將 Unix 時間轉換為 datetime 物件"""
    return datetime.fromtimestamp(timestamp)


def extract_time_list_data(files):
    
    # initialize time stamp list
    data_time_list, torq_time_list, spd_time_list, pwr_time_list, eff_time_list, acc_time_list = [], [], [], [], [], [],
    first_time = None

    for file in files:
        data_read = rul_rd.read_rul_data(file)  # 這裡假設檔案是 CSV 格式

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


    return basic_extract_result


def plot_basic_time_list(basic_extract_result):
    plt.figure(figsize=(6, 5))

    # Subplot for Torque
    plt.subplot(3, 2, 1)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["torque_time_list"], linestyle='-')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Torque")
    plt.title("Torque vs Time Stamps")
    plt.grid()

    # Subplot for Speed
    plt.subplot(3, 2, 2)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["speed_time_list"], linestyle='-')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Speed")
    plt.title("Speed vs Time Stamps")
    plt.grid()

    # Subplot for Power
    plt.subplot(3, 2, 3)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["power_time_list"], linestyle='-')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Power")
    plt.title("Power vs Time Stamps")
    plt.grid()

    # Subplot for Efficiency
    plt.subplot(3, 2, 4)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["efficiency_time_list"], linestyle='-')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Time Stamps")
    plt.grid()

    # Subplot for Vibration
    plt.subplot(3, 2, 5)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["vibration_time_list"], linestyle='-')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Vibration")
    plt.title("Vibration vs Time Stamps")
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_basic_time_slice(file, basic_extract_result, idx):
    sampled_data = rul_rd.read_rul_data(file)
    fig = plt.figure(figsize=(20, 24))

    # Subplot for voltage
    plt.subplot(6, 1, 1)
    plt.plot(sampled_data["Voltage alpha"], linestyle='-')
    plt.plot(sampled_data["Voltage beta"], linestyle='-')
    plt.xlabel("sample point")
    plt.ylabel("[V]")
    plt.title("Voltage")
    plt.grid()

    # Subplot for Current
    plt.subplot(6, 1, 2)
    plt.plot(sampled_data["Current alpha"], linestyle='-')
    plt.plot(sampled_data["Current beta"], linestyle='-')
    plt.xlabel("sample point")
    plt.ylabel("[A]")
    plt.title("Current")
    plt.grid()

    # Subplot for Vibration
    plt.subplot(6, 1, 3)
    plt.plot(sampled_data["vibration data"], linestyle='-')
    plt.xlabel("sample point")
    plt.ylabel("[g]")
    plt.title("Vibration")
    plt.grid()

    # Subplot for Efficiency
    plt.subplot(6, 1, 4)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["vibration_time_list"], linestyle='-')
    plt.plot(basic_extract_result["Time stamps"][idx], basic_extract_result["vibration_time_list"][idx], 'ro', label='Sampled Point')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Time Stamps")
    plt.grid()

    # Subplot for Torque
    plt.subplot(6, 1, 5)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["torque_time_list"], linestyle='-')
    plt.plot(basic_extract_result["Time stamps"][idx], basic_extract_result["torque_time_list"][idx], 'ro', label='Sampled Point')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Torque")
    plt.title("Torque vs Time Stamps")
    plt.grid()

    # Subplot for Speed
    plt.subplot(6, 1, 6)
    plt.plot(basic_extract_result["Time stamps"], basic_extract_result["speed_time_list"], linestyle='-')
    plt.plot(basic_extract_result["Time stamps"][idx], basic_extract_result["speed_time_list"][idx], 'ro', label='Sampled Point')
    plt.xlabel("Time Stamps (minutes)")
    plt.ylabel("Speed")
    plt.title("Speed vs Time Stamps")
    plt.grid()

    plt.tight_layout()
    return fig


def create_rul_animations(files, motor_time_list, mp4_path, html_path, fps=10, duration=5, title="RUL Prediction Animation", xlabel="Time (minutes)", ylabel="RUL", xlim=(0, 100), ylim=(0, 100)):
    """
    Create an animation of RUL prediction and save it as an HTML file.
    
    Parameters:
        files (list): List of file paths to the data files.
        motor_time_list (list): List of time data for the motor.
        mp4_path (str): Path to save the MP4 file.
        html_path (str): Path to save the HTML file.
        fps (int): Frames per second for the animation.
        duration (int): Duration of the animation in seconds.
        title (str): Title of the animation.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
    """
    
    import matplotlib as mpl
    mpl.rcParams['animation.embed_limit'] = 50 * 1024 * 1024  
    
    # 建立子圖物件
    fig, axs = plt.subplots(6,1, figsize=(10, 12), dpi=80)
    
    # 初始化子圖信號
    Voltage_line_alpha, = axs[0].plot([], [], label='Voltage Alpha', color='blue')
    Voltage_line_beta, = axs[0].plot([], [], label='Voltage Beta', color='red')
    Current_line_alpha, = axs[1].plot([], [], label='Current Alpha', color='blue')
    Current_line_beta, = axs[1].plot([], [], label='Current Beta', color='red')
    Vibration_line, = axs[2].plot([], [], label='Vibration', color='blue')
    Vibration_rms_line, = axs[3].plot(motor_time_list["Time stamps"], motor_time_list["vibration_time_list"], label='Vibration rms', color='blue')
    Current_rms_point, = axs[3].plot([], [], 'ro', label='Sampled Point')
    Torque_line, = axs[4].plot(motor_time_list["Time stamps"], motor_time_list["torque_time_list"], label='Torque', color='blue')
    Current_torq_point, = axs[4].plot([], [], 'ro', label='Sampled Point')
    Speed_line, = axs[5].plot(motor_time_list["Time stamps"], motor_time_list["speed_time_list"], label='Speed', color='blue')
    Current_speed_point, = axs[5].plot([], [], 'ro', label='Sampled Point')
    
    
    # 初始化子圖設定
    axs[0].set_title("Voltage")
    axs[0].set_xlabel("sample point")
    axs[0].set_ylabel("[V]")
    axs[0].grid()
    axs[1].set_title("Current") 
    axs[1].set_xlabel("sample point")
    axs[1].set_ylabel("[A]")
    axs[1].grid()
    axs[2].set_title("Vibration")
    axs[2].set_xlabel("sample point")
    axs[2].set_ylabel("[g]")
    axs[2].grid()
    axs[3].set_title("Vibration rms ")
    axs[3].set_xlabel("Time Stamps (minutes)")
    axs[3].set_ylabel("[g]]")
    axs[3].grid()
    axs[4].set_title("Torque")
    axs[4].set_xlabel("Time Stamps (minutes)")
    axs[4].set_ylabel("Torque")
    axs[4].grid()
    axs[5].set_title("Speed")
    axs[5].set_xlabel("Time Stamps (minutes)")
    axs[5].set_ylabel("Speed")
    axs[5].grid()
    
    
    # 設定 x 軸和 y 軸的範圍
    data_read = rul_rd.read_rul_data(files[0])
    axs[0].set_xlim(0, len(data_read["Voltage alpha"]))
    axs[1].set_xlim(0, len(data_read["Current alpha"]))
    axs[2].set_xlim(0, len(data_read["vibration data"]))
    axs[3].set_xlim(min(motor_time_list["Time stamps"]), max(motor_time_list["Time stamps"])) 
    axs[4].set_xlim(min(motor_time_list["Time stamps"]), max(motor_time_list["Time stamps"]))
    axs[5].set_xlim(min(motor_time_list["Time stamps"]), max(motor_time_list["Time stamps"]))
    
    axs[0].set_ylim(-50, 50)
    axs[1].set_ylim(min(data_read["Current alpha"])-0.1, max(data_read["Current alpha"]+0.1))
    axs[2].set_ylim(-1, 1)
    axs[3].set_ylim(min(motor_time_list["vibration_time_list"]), max(motor_time_list["vibration_time_list"]))
    axs[4].set_ylim(min(motor_time_list["torque_time_list"]), max(motor_time_list["torque_time_list"]))
    axs[5].set_ylim(min(motor_time_list["speed_time_list"]), max(motor_time_list["speed_time_list"]))
    
    fig.tight_layout()
    
    # 預讀取所有數據
    all_data = [rul_rd.read_rul_data(f) for f in files]

    
    def update(frame):
        # 讀取數據
        # data_read = rul_rd.read_rul_data(files[frame])
        data_read = all_data[frame]
        # 更新子圖數據
        Voltage_line_alpha.set_data(range(len(data_read["Voltage alpha"])), data_read["Voltage alpha"])
        Voltage_line_beta.set_data(range(len(data_read["Voltage beta"])), data_read["Voltage beta"])
        Current_line_alpha.set_data(range(len(data_read["Current alpha"])), data_read["Current alpha"])
        Current_line_beta.set_data(range(len(data_read["Current beta"])), data_read["Current beta"])
        Vibration_line.set_data(range(len(data_read["vibration data"])), data_read["vibration data"])
        # Vibration_rms_line.set_data(motor_time_list["Time stamps"], motor_time_list["vibration_time_list"])
        Current_rms_point.set_data([motor_time_list["Time stamps"][frame]], [motor_time_list["vibration_time_list"][frame]])
        # Torque_line.set_data(motor_time_list["Time stamps"], motor_time_list["torque_time_list"])
        Current_torq_point.set_data([motor_time_list["Time stamps"][frame]], [motor_time_list["torque_time_list"][frame]])
        # Speed_line.set_data(motor_time_list["Time stamps"], motor_time_list["speed_time_list"])
        Current_speed_point.set_data([motor_time_list["Time stamps"][frame]], [motor_time_list["speed_time_list"][frame]])

        
        return Voltage_line_alpha, Voltage_line_beta, Current_line_alpha, Current_line_beta, Vibration_line, Current_rms_point, Current_torq_point, Current_speed_point
      
    # Create HTML animation
    ani_html = animation.FuncAnimation(fig, update, frames=len(files), blit=True, interval=1000//fps)
    # ani_html = animation.FuncAnimation(fig, update, frames=10, blit=True, interval=1000//fps)
    html_animation = HTML(ani_html.to_jshtml())
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(html_animation.data)
        
    # Create MP4 animation
    # writer_mp4 = animation.FFMpegWriter(fps=fps, bitrate=2000)
    # ani_mp4 = animation.FuncAnimation(fig, update, frames=len(files), blit=True)
    # ani_mp4.save(mp4_path, writer=writer_mp4)
        
    return 

if __name__ == "__main__":
    
    #%%
    # 合併主資料夾與子資料夾路徑
    NTU_dataset_path = r'D:\OneDrive\RUL HI Reasearch Result\Data_sets\NTU_RUL_v2_data\Acc_life_test_data\Organized_Data'
    
    # collect_setting = r"Load_Free\Speed_1800\Pressure_10psi"  
    # collect_name=r"0312_V1800_10P_FREE0"
    # collect_name=r"0313_V1800_10P_FREE1"
    # collect_name=r"0314_V1800_10P_FREE2"
    
    # collect_setting = r"Load_1V\Speed_1800\Pressure_10psi"
    # collect_name=r"10kg_1V_1800rpm_1"
    # collect_name=r"15kg_1V_1800rpm_1"
    # collect_name=r"15kg_1V_1800rpm_2"
    
    
    collect_setting = r"Load_1V\Speed_1800\Pressure_5psi"
    # collect_name=r"5psi_1V_1_3min"
    collect_name=r"06kg_1V_1800rpm_2"
  
    
    
    
    folder_path = os.path.join(NTU_dataset_path, collect_setting, collect_name)
  
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    # sort file by file number 
    files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
            

    if files:
        motor_time_list=extract_time_list_data(files)
        print('extract time list data done')
    else:
        print("未找到 CSV 檔案！")

    fig=plot_basic_time_slice(files[0], motor_time_list, 0)
    
    #%%
    html_path = os.path.join(".", "animations", collect_name+"_history.html")
    mp4_path = os.path.join(".", "animations", "rul_prediction_animation.mp4")
    time_now=time.time()
   
    create_rul_animations(
        files, 
        motor_time_list,
        mp4_path=mp4_path, 
        html_path=html_path,
        fps=10,
        duration=5,
        title="RUL Prediction Animation",
        xlabel="Time (minutes)",
        ylabel="RUL",
    )
    
    print('aninmation generation time:', time.time()-time_now)
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    display(HTML(html_content))


    #%%
