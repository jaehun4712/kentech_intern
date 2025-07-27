import pandas as pd
import matplotlib.pyplot as plt


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

# 방전 구간 (주행만)
df_drive = df[
    (df['pack_current'] > 0) &
    (df['chrg_cable_conn'] == 0)&
   (df['speed'] > 0)
].copy()    #”세가지 조건을 모두 만족 하는 구간에서의 pack_power_watt(전력) 데이터만 이용하여 주행구간에서 총 소비된 에너지(Wh) 계산”

df_drive.loc[:, 'energy_Wh'] = df_drive['pack_power_watt'] * df_drive['delta_sec'] / 3600

#충전구간(주차+충전기o)
df_charging=df[
    (df['chrg_cable_conn']==1)&
    (df['speed']==0)
].copy()

df_charging.loc[:, 'energy_Wh']=df_charging['pack_power_watt']*df_charging['delta_sec']/3600


df['is_regen'] = (                                  # 회생제동인지 아닌지 여부를 True/False로 판단
    (df['pack_current'] < 0) &
    (df['chrg_cable_conn'] == 0) &
    (df['speed'] > 0)
) # 회생제동이 일어나는 구간조건을 설정하여 회생제동시 True값을 반환함. is_region은 True, False로 이루어진 열.

df['regen_group'] = (df['is_regen'] != df['is_regen'].shift()).cumsum()
#is_regen의 열과 한칸 shift하여 얻게된 열을 비교하여 그 값이 다르면 .cumsum()에 의해 카운트. 카운트된 숫자가 변할때 마다
#회생->주행->회생이 변하는 구간이 된다.

regen_groups = df[df['is_regen']].groupby('regen_group')
#df['is_regen']은 True/False로 이루어진 행. df[df['is_regen']]은 True행만 남기고 False는 걸러서 제거.
#즉, True 상태인 회생제동 구간만 그룹화 진행.
#Boolean 타입. 조건문 역할. True/False로 이루어져있을 때 True값만 얻어내겠다.
#뒤에 .groupby('regen_group')를 붙여주면, regen_groups로 그 값들을 묶겠다는 뜻.
#결론적으로 회생제동 구간만 << df['is_regen']에서 True인 구간만>> 선정하여 그룹화 한다는 뜻.
valid_regen_indices = []

# 시간 조건 필터링: 회생이 1초 이상 지속되는 구간만 인정
for group_id, group in regen_groups:   #group_id = 그룹명/regen group의 숫자/회생구간번호, group : 데이터프레임
    #for..in.. 그룹 하나하나에 대한 반복수행
    duration = group['delta_sec'].sum() #group은 하나의 회생제동에 대한 데이터. delt sec를 통해 각 회생구간 샘플간의 시간 간격
    # .sum()을 통해 시간간격을 모두 더해주면 그 회생제동구간의 전체 지속시간.
    if duration >= 1:  # 1초 이상 지속되는 회생만 유효
        valid_regen_indices.extend(group.index)
        #.extend() 리스트에 여러값을 한꺼번에 추가하는 함수.
        # group.index=그 구간의 행번호(인덱스). 즉 1초이상의 조건을 만족하는 회생구간의 행번호(인덱스)를
        #valid_regen_indices에 저장.

df['is_valid_regen'] = df.index.isin(valid_regen_indices)
#

# 최종 회생 데이터셋
df_regen = df[df['is_valid_regen']].copy()
df_regen['energy_Wh'] = df_regen['pack_power_watt'] * df_regen['delta_sec'] / 3600
# 방전 에너지 총합 (양수)
total_discharge_Wh = df_drive['energy_Wh'].sum()

#충전에너지총합 ... 음수라서 부호 반전
total_charge_Wh=-df_charging['energy_Wh'].sum()

# 회생 에너지 총합 (음수이므로 절댓값 취함)
total_regen_Wh = -df_regen['energy_Wh'].sum()  # 음수라서 부호 반전

# 순 소비 에너지 (방전 - 회생)
net_energy_Wh = total_discharge_Wh - total_regen_Wh
# 충전에너지(충전+회생)
net_chrg_energy_Wh=total_charge_Wh + total_regen_Wh

# 주행 거리
start_km = df['odometer'].iloc[0] # 첫번째행
end_km = df['odometer'].iloc[-1]  #마지막행
total_distance_km = end_km - start_km #odometer의 마지막 행 - 첫번째행 == 주행거리!

if total_distance_km > 0:
    ev_efficiency = net_energy_Wh / total_distance_km
    ev_efficiency2= 100*net_energy_Wh/net_chrg_energy_Wh
    print(f"총 방전 에너지: {total_discharge_Wh:.2f} Wh")
    print(f"총 회생 에너지: {total_regen_Wh:.2f} Wh")
    print(f"순 소비 에너지: {net_energy_Wh:.2f} Wh")
    print(f"총 충전 에너지: {net_chrg_energy_Wh:.2f}Wh")
    print(f"총 주행 거리: {total_distance_km:.2f} km")
    print(f"EV 효율 (순 소비 에너지 기준): {ev_efficiency:.2f} Wh/km")
    print(f"EV 효율2(순소비에너지용량/충전에너지용량 *100%): {ev_efficiency2:.2f} %")
else:
    print("주행거리가 0이므로 효율 계산 불가")

plt.figure(figsize=(14, 6))
plt.plot(df['time'], df['pack_current'], label='Pack Current (A)', color='blue', alpha=0.5)
plt.plot(df['time'], df['speed'], label='Speed (km/h)', color='green', alpha=0.5)
plt.title('Pack Current & Vehicle Speed over Time')
plt.xlabel('Time')
plt.ylabel('Current / Speed')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df_drive['time'], df_drive['pack_current'], label='Discharge Current (A)', color='red')
plt.plot(df_regen['time'], df_regen['pack_current'], label='Regen Current (A)', color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Pack Current During Drive and Regeneration')
plt.xlabel('Time')
plt.ylabel('Pack Current (A)')
plt.legend()
plt.tight_layout()
plt.show()

df_drive['cumulative_energy_Wh'] = df_drive['energy_Wh'].cumsum()
plt.figure(figsize=(14, 6))
plt.plot(df_drive['time'], df_drive['cumulative_energy_Wh'], label='Cumulative Energy (Wh)', color='purple')
plt.title('Cumulative Energy Consumption over Time')
plt.xlabel('Time')
plt.ylabel('Energy (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df['time'], df['pack_current'], label='Pack Current', color='black')

# 회생제동 구간 강조
for i in range(len(df_regen)):
    plt.axvspan(df_regen['time'].iloc[i],
                df_regen['time'].iloc[i] + pd.Timedelta(seconds=df_regen['delta_sec'].iloc[i]),
                color='lightblue', alpha=0.3)

plt.title('Pack Current with Regen Zones Highlighted')
plt.xlabel('Time')
plt.ylabel('Current (A)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df_regen['pack_current'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Regen Current (A)')
plt.xlabel('Pack Current (A)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 회생 구간 그룹별 지속 시간 계산 (regen_group은 앞서 정의된 회생 구간 그룹 번호)
regen_groups = df[df['is_valid_regen']].groupby('regen_group')
regen_durations = [group['delta_sec'].sum() for _, group in regen_groups]

plt.figure(figsize=(8, 5))
plt.hist(regen_durations, bins=20, color='orange', edgecolor='black')
plt.title('Regen Duration per Group')
plt.xlabel('Regen Duration (seconds)')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_drive['pack_power_watt'], bins=60, color='red', alpha=0.6, label='Discharge')
plt.hist(df_regen['pack_power_watt'], bins=60, color='blue', alpha=0.6, label='Regen')
plt.title('Pack Power Distribution: Discharge vs Regen')
plt.xlabel('Pack Power (Watt)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df['time'], df['pack_current'], label='Pack Current', color='gray', alpha=0.4)
plt.scatter(df[df['is_valid_regen']]['time'],
            df[df['is_valid_regen']]['pack_current'],
            color='blue', label='Regen Current', s=10)
plt.axhline(0, color='black', linestyle='--')
plt.title('Pack Current with Regen Periods Highlighted')
plt.xlabel('Time')
plt.ylabel('Current (A)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(df['time'], df['pack_current'], color='gray', label='Pack Current', alpha=0.5)
plt.scatter(df[df['is_valid_regen']]['time'],
            df[df['is_valid_regen']]['pack_current'],
            color='blue', label='Regen Current', s=10)
plt.axhline(0, color='black', linestyle='--')
plt.title("Regen Current Highlighted Over Time")
plt.legend()
plt.tight_layout()
plt.show()