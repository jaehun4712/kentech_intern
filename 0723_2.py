
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

# ----------------------------
# 파일 경로
# ----------------------------
bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\3week_data\bms_01241225206_2023-08.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

# ----------------------------
# BMS 데이터 로드 및 시간 처리
# ----------------------------
bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], format='mixed', errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)

# delta_sec 계산 및 segment 구분
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)
gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()#시간간격이 60초보다 크면 bms꺼짐 or 날짜변경으로
                                                                   #판단하여 단절지점 분류

segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments

# 각 세그먼트에서 다시 delta_sec 계산
bms_df['delta_sec'] = (
    bms_df.groupby('segment_id')['time']#그룹화한 segment_id내에서 시간간격 계산
    .diff()
    .dt.total_seconds() #시간간격을 초단위로 변경
    .fillna(0)
)

# delta_sec > 10 제거용 마스크
bms_df['valid'] = bms_df['delta_sec'] <= 10

# ----------------------------
# OCV 곡선 처리
# ----------------------------
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]] #C_rate=0.05C로 수정
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)

soc_vals = soc_ocv_data['SOC'].values/100
ocv_vals = soc_ocv_data['OCV'].values

valid_mask = (~np.isnan(soc_vals)) & (~np.isnan(ocv_vals))
soc_vals = soc_vals[valid_mask]
ocv_vals = ocv_vals[valid_mask]

sort_idx = np.argsort(soc_vals)
soc_vals = soc_vals[sort_idx]
ocv_vals = ocv_vals[sort_idx]

ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value="extrapolate")

# ----------------------------
# SOC 기반 에너지 계산
# ----------------------------
Qmax = 56.47*192*2

bms_columns = [col.lower() for col in bms_df.columns]
soc_col = next((col for col in bms_columns if 'soc' in col), None)

if soc_col:
    soc_series = bms_df.loc[bms_df['valid'], bms_df.columns[bms_columns.index(soc_col)]].dropna()
    soc_0 = soc_series.iloc[0]/100
    soc_1 = soc_series.iloc[-1]/100

    def compute_estored(soc):
        if soc <= 0.001:
            return 0
        try:
            result, _ = quad(ocv_func, 0, soc, limit=200)
            return Qmax * result
        except Exception as e:
            print(f"[적분 실패] SOC: {soc:.4f} → 오류: {e}")
            return 0

    e_stored_0 = compute_estored(soc_0)
    e_stored_1 = compute_estored(soc_1)
    e_stored_diff = e_stored_0 - e_stored_1

    print(f"SOC_0: {soc_0 * 100:.1f}%, E_stored_0: {e_stored_0:.2f} Wh")
    print(f"SOC_1: {soc_1 * 100:.1f}%, E_stored_1: {e_stored_1:.2f} Wh")
    print(f"E_stored_0 - E_stored_1: {e_stored_diff:.2f} Wh")
else:
    print("⚠️ SOC 컬럼을 찾을 수 없습니다.")
    exit()

# ----------------------------
# 조건별 에너지 계산
# ----------------------------
df = bms_df.copy()
df.columns = [col.lower() for col in df.columns]

cable_col = 'chrg_cable_conn'
speed_col = 'speed'
current_col = 'pack_current'
voltage_col = 'pack_volt'
time_interval = 'delta_sec'

def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600

# 조건 정의 (모두 valid 구간만 포함)
cond_discharge_drive = df['valid'] & (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)
cond_discharge_idle  = df['valid'] & (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)
cond_trip_charge     = df['valid'] & (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] < 0)
cond_charging        = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)
cond_charging_idle   = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)

useless_possible = (df['valid']&(df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] == 0)) |(df['valid']&(df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] == 0))
useless_impossible = (df['valid']&(df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] < 0)) |(df['valid']&(df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] == 0))



# 에너지 계산
E_trip_discharge_drive = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_discharge_drive].sum()
E_trip_discharge_idle  = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_discharge_idle].sum()
E_trip_discharge = E_trip_discharge_drive + E_trip_discharge_idle

E_trip_charge = calc_energy(df[voltage_col], -df[current_col], df[time_interval])[cond_trip_charge].sum()
E_real_charging = calc_energy(df[voltage_col], -df[current_col], df[time_interval])[cond_charging].sum()
E_charging_idle = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_charging_idle].sum()
E_charging = E_real_charging - E_charging_idle
E_trip_net = E_trip_discharge - E_trip_charge

E_useless_possible=calc_energy(df[voltage_col], df[current_col], df[time_interval])[useless_possible].sum()
E_useless_impossible=calc_energy(df[voltage_col], -df[current_col], df[time_interval])[useless_impossible].sum()
E_useless=E_useless_possible+E_useless_impossible

# ----------------------------
# 출력
# ----------------------------
print(f".................................")
print(f"E_trip_discharge_drive: {E_trip_discharge_drive:.2f} Wh")
print(f"E_trip_discharge_idle: {E_trip_discharge_idle:.2f} Wh")
print(f"E_trip_discharge = E_trip_discharge_drive + E_trip_discharge_idle: {E_trip_discharge:.2f} Wh")
print(f"E_trip_charge (regen): {E_trip_charge:.2f} Wh")
print(f"E_trip_net = E_trip_discharge - E_trip_charge (regen) : {E_trip_net:.2f} Wh")
print(f".................................")
print(f"E_real_charging : {E_real_charging:.2f} Wh")
print(f"E_charging_idle : {E_charging_idle:.2f} Wh")
print(f"E_charging = E_real_charging - E_charging_idle: {E_charging:.2f} Wh")
print(f".................................")

if (E_charging + e_stored_diff) > 0:
    efficiency1 = E_trip_net / (E_charging + e_stored_diff) * 100
    print(f"Efficiency e1 = {efficiency1:.4f}%")

    efficiency2 = (E_trip_net + e_stored_1) / (E_charging + e_stored_0) * 100
    print(f"Efficiency e2 = {efficiency2:.4f}%")
    print(f".................................")
else:
    print("⚠️ Efficiency 계산 불가: 분모가 0 또는 음수")

import matplotlib.pyplot as plt

# ----------------------------
# 계산된 에너지 값 (앞에서 계산된 결과 그대로 사용)
# ----------------------------
# 예: 이 변수들은 앞 코드에서 이미 계산된 값임
# E_trip_discharge_drive
# E_trip_charge
# E_trip_discharge_idle
# E_real_charging
# E_charging_idle

# 에너지 항목 딕셔너리 구성
energy_parts = {
'Totatl Energy': E_trip_discharge_drive+ E_trip_charge+E_trip_discharge_idle+E_real_charging+ E_charging_idle,
    'Useless Energy' : E_useless
}

# 전체 에너지 합
total_energy = sum(energy_parts.values())

# 비율 및 라벨 구성
labels = [f"{k}\n({v:.2f} Wh)" for k, v in energy_parts.items()]
sizes = [v / total_energy * 100 for v in energy_parts.values()]

# 시각화
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%.4f%%', startangle=140)
plt.title("Energy Components Ratio")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
# 조건별 마스크 계산
cond1 = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)#(1)주행만
cond2 = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] < 0)#(2)주행 중 회생제동
cond3 = (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)#(3)정지+충전기x+대기전력소모
cond4 = (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)#(4)정지+충전기o
cond5 = (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)#(5)정지+충전기o+대기전력소모


cond6 = (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] == 0)#가능) 완전한 정지상태. 충전기x, speed=0, current=0
cond7 = (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] == 0)#가능) 완전한 정지상태. 충전기연결, speed=0, current=0

cond8 = (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] <0)# 왜나오지.?? 충전기 없이 speed=0이지만 current가 음수인경우
cond9 = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] == 0)# 왜나오지..?? 충전기 없이 speed있지만 current 없는 경우


#cond10 = (df[cable_col] == 1) & (df[speed_col] > 0)  # 이론상 불가능+bms에도 존재 안하는부분

combined = cond1 | cond2 | cond3 | cond4 | cond5| cond6| cond7 |cond8| cond9
# 7가지상황+ㅈmissing상황의 합집합. missing data가 더 있는지 추가점검

# 빠진 구간 마스크
missing = ~combined
plt.figure(figsize=(15,3))
plt.plot(missing.index, missing.astype(int), label='Missing data (Not in 9 parts)', color='green')
#plt.plot(cond1.index, cond1.astype(int), label='Discharge Drive', alpha=0.7)
#plt.plot(cond2.index, cond2.astype(int), label='Charge Regen', alpha=0.7)
#plt.plot(cond3.index, cond3.astype(int), label='Discharge Idle', alpha=0.7)
#plt.plot(cond4.index, cond4.astype(int), label='Real Charging', alpha=0.7)
#plt.plot(cond5.index, cond5.astype(int), label='Charging Idle', alpha=0.7)
#plt.plot(cond6.index, cond6.astype(int), label='0 0 0', alpha=0.7, color='black')
#plt.plot(cond7.index, cond7.astype(int), label='1 0 0', alpha=0.7, color='black')
#plt.plot(cond8.index, cond8.astype(int), label='0 0 -', color='blue')
#plt.plot(cond9.index, cond9.astype(int), label='0 + 0', color='yellow')


#plt.plot(cond10.index, cond10.astype(int), label='cable=1, speed + ', color='gray')

plt.yticks([0,1])
plt.xlabel('Index or Time')
plt.title('Condition Coverage Over Time')
plt.legend(loc='upper right')
plt.show()



