import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기준
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지



import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.plistlib import end_string

# 1. 파일 불러오기
file_path = r"C:\Users\OWNER\Desktop\EV6_2301\0424raw\bms_01241124056_2024-04-24.csv" #20240424의 하루 raw data
df = pd.read_csv(file_path)

# 2. 시간 컬럼 변환
df['time'] = pd.to_datetime(df['time']) # 데이터를 문자열에서 datetime 자료형으로 바꿔주어야, 시간순 정렬과 간격계산 가능.
df = df.sort_values('time') # 시간순으로 정렬

# 3. 속도 컬럼 지정 (데이터에 따라 'emobility_spd' 또는 다른 것 사용)
df['speed'] = df['emobility_spd']  # embility_spd의 열이름을 speed로 수정. 단위:km

# 4. 전력 및 시간 간격 계산
df['pack_power_watt'] = df['pack_volt'] * df['pack_current']
# pack_volt 열과 pack_current의 값을 곱하여 pack_power_watt로 새로운 열을 만듦.

df['delta_sec'] = df['time'].diff().dt.total_seconds().fillna(0)
#(1).diff()를 이용하여 time열의 간격 계산 (2).dt.total_seconds()를 이용하여 시간간격을 초단위로 변환
# 3).fillna(0)을 이용해 비어있는 첫행을 0으로 채워주어 시간간격 계산

#E_trip_discharging (주행만)
df_drive = df[
     (df['chrg_cable_conn'] == 0)&
     (df['speed'] > 0)&
     (df['pack_current'] > 0)
].copy()
df_drive.loc[:, 'drive_Wh'] = df_drive['pack_power_watt'] * df_drive['delta_sec'] / 3600

#E_trip_discharging(주차+충전x+대기전력소모)
df_prk_nochrg=df[
    (df['chrg_cable_conn']==0)&
    (df['speed']==0)&
    (df['pack_current']>0)
].copy()
df_prk_nochrg.loc[:, 'prk_nochrg_Wh'] = df_prk_nochrg['pack_power_watt'] * df_prk_nochrg['delta_sec'] / 3600

#E_trip_charging(주행 중 break : 회생제동)
df_regen = df[
     (df['chrg_cable_conn'] == 0)&
     (df['speed'] > 0)&
     (df['pack_current'] < 0)
].copy()    #”세가지 조건을 모두 만족 하는 구간에서의 pack_power_watt(전력) 데이터만 이용하여 주행구간에서 총 소비된 에너지(Wh) 계산”
df_regen.loc[:, 'regen_Wh'] = df_regen['pack_power_watt'] * df_regen['delta_sec'] / 3600


#E_charging_charging(주차+충전기o)
df_charging=df[
    (df['chrg_cable_conn']==1)&
    (df['speed']==0)&
    (df['pack_current']<0)
].copy()
df_charging.loc[:, 'charging_Wh']=df_charging['pack_power_watt']*df_charging['delta_sec']/3600

#E_charging_discharging(주차+충전기o+완충(socd=100%)+대기전력소모)
df_charging_ps=df[
    (df['chrg_cable_conn']==1)&
    (df['speed']==0)&
    (df['pack_current']>0)
].copy()
df_charging_ps.loc[:, 'charging_ps_Wh']=df_charging_ps['pack_power_watt']*df_charging_ps['delta_sec']/3600


# 방전 에너지 총합 = (주행) + (주차, 충전기x, 대기전력소모)
total_discharge_Wh = df_drive['drive_Wh'].sum()+df_prk_nochrg['prk_nochrg_Wh'].sum()

# 회생 에너지 총합 (음수이므로 절댓값 취함)
total_regen_Wh = -df_regen['regen_Wh'].sum()  # 음수라서 부호 반전

# 충전에너지총합 (음수이므로 절댓값)
total_charge_Wh=-df_charging['charging_Wh'].sum()

# 충전 중 대기전력소모에너지
total_charge_ps_Wh=df_charging_ps['charging_ps_Wh'].sum()


# 주행 중 순 소비 에너지 (방전) - (회생)
net_trip_energy_Wh = total_discharge_Wh-total_regen_Wh

# 충전 에너지 =  (충전 에너지) - (충전 중 대기전력소모 에너지)
net_charge_energy_Wh=total_charge_Wh-total_charge_ps_Wh


ev_efficiency= 100*net_trip_energy_Wh/net_charge_energy_Wh

print(f"총 주행 중 방전 에너지: {total_discharge_Wh:.2f} Wh")
print(f"총 주행 중 회생 에너지: {total_regen_Wh:.2f} Wh")
print(f"순 소비 에너지: {net_trip_energy_Wh:.2f} Wh")
print(f"총 충전 에너지: {net_charge_energy_Wh:.2f}Wh")
print(f"EV 효율 (순소비에너지용량/충전에너지용량 *100%): {ev_efficiency:.2f} %")

import matplotlib.pyplot as plt

# 전체 주행 데이터 시각화를 위해 전체 시간 대비 pack_current 플롯
plt.figure(figsize=(15, 5))
plt.plot(df['time'], df['pack_current'], label='Pack Current', color='gray', alpha=0.5)

# 회생제동 구간 강조 (pack_current < 0)
plt.scatter(df_regen['time'], df_regen['pack_current'], color='red', label='Regenerative Braking', s=10)

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Pack Current with Regenerative Braking Highlighted")
plt.xlabel("Time")
plt.ylabel("Pack Current (A)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#속도 + 회생구간
plt.figure(figsize=(15, 5))
plt.plot(df['time'], df['speed'], label='Speed (km/h)', color='blue', alpha=0.6)
plt.scatter(df_regen['time'], df_regen['speed'], color='green', label='Speed during Regen', s=10)
plt.title("Speed Profile with Regenerative Braking Points")
plt.xlabel("Time")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#soc+회생구간

fig, ax1 = plt.subplots(figsize=(15, 5))

# 첫 번째 y축: Pack Current (회생제동)
ax1.plot(df['time'], df['pack_current'], label='Pack Current (A)', color='gray', alpha=0.4)
ax1.scatter(df_regen['time'], df_regen['pack_current'], color='red', label='Regen Current', s=10)
ax1.set_ylabel('Pack Current (A)', color='red')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.tick_params(axis='y', labelcolor='red')

# 두 번째 y축: SOC
ax2 = ax1.twinx()
ax2.plot(df['time'], df['soc'], label='SOC (%)', color='blue')
ax2.set_ylabel('SOC (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# 제목 및 범례
fig.suptitle("Regenerative Braking & SOC Change Over Time")
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
ax1.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# 회생제동 구간: 속도 > 0, 충전 케이블 X, 전류 < 0
df_regen = df[
    (df['chrg_cable_conn'] == 0) &
    (df['speed'] > 0) &
    (df['pack_current'] < 0)
].copy()

# 시간 순 정렬 (혹시 모르니)
df_regen = df_regen.sort_values('time')

# 그래프 그리기
plt.figure(figsize=(12, 5))
plt.plot(df_regen['time'], df_regen['soc'], label='SOC during Regen', color='green')
plt.xlabel('Time')
plt.ylabel('SOC (%)')
plt.title('SOC 변화 (회생제동 구간)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
