import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일들이 있는 폴더 경로
folder_path = r"C:\Users\OWNER\Desktop\data"

# 해당 폴더 내 모든 CSV 파일 경로 리스트 생성
trip_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

# Trip별로 따로 그리기
for file_path in trip_files:
    try:
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])

        # Figure 생성
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Trip Data: {os.path.basename(file_path)}", fontsize=14)

        axs[0].plot(df['time'], df['soc'], color='black', marker='o', linestyle='None', markersize=3)
        axs[0].set_ylabel('SOC (%)')
        axs[0].set_title('SOC vs. Time')
        axs[0].grid(True)

        axs[1].plot(df['time'], df['pack_current'], color='blue')
        axs[1].set_ylabel('Pack Current (A)')
        axs[1].set_title('Pack Current vs. Time')
        axs[1].grid(True)

        axs[2].plot(df['time'], df['pack_volt'], color='red')
        axs[2].set_ylabel('Pack Voltage (V)')
        axs[2].set_title('Pack Voltage vs. Time')
        axs[2].set_xlabel('Time')
        axs[2].tick_params(axis='x', rotation=45)
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"[에러] {file_path}: {e}")
