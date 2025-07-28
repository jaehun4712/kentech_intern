import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
file_path = r"C:\Users\OWNER\Desktop\EV6_2301\0424raw\bms_01241124056_2024-04-24.csv"

# 1. 데이터 로드 및 시간 처리
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# 2. 컬럼명 확인
print(df.columns.tolist())  # 여기서 정확한 speed 관련 열 이름 확인 필요

# 3. 속도 및 필요한 컬럼 지정 (예: 'emobility_spd' 사용)
df['speed'] = df['emobility_spd']  # 또는 다른 속도 열 사용

# 4. 속도 변화량 및 회생제동 여부 계산
df['speed_delta'] = df['speed'].diff().fillna(0)

df['is_regen'] = (
    (df['pack_current'] < -5) &                     # -5A 이하 전류
    (df['chrg_cable_conn'] == 0) &                  # 충전기 연결 X
    (df['speed'] > 10) &                            # 속도 조건
    (df['speed_delta'] < -0.5)                      # 감속 중
)

# 5. 회생제동 시각화
plt.figure(figsize=(14, 6))
plt.plot(df['time'], df['pack_current'], label='Pack Current (A)', color='blue')
plt.fill_between(df['time'], df['pack_current'], where=df['is_regen'],
                 color='green', alpha=0.3, label='Regen Braking Detected')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Regen Braking Detection with Noise Filtering')
plt.xlabel('Time')
plt.ylabel('Pack Current (A)')
plt.legend()
plt.tight_layout()
plt.show()
