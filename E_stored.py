import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

# 파일 경로
bms_file = r"C:\Users\OWNER\Desktop\Ionic5_7_bmsdata\bms_01241227999_2023-07.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

# BMS 데이터 로드
bms_df = pd.read_csv(bms_file)

# OCV 곡선 시트 로드 및 전처리
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 1] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [1, 2]]
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)

# OCV 보간 함수 정의 (SOC는 0~1로 스케일링)
soc_vals = soc_ocv_data['SOC'].values / 100
ocv_vals = soc_ocv_data['OCV'].values
ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value="extrapolate")

# Qmax (엑셀 시트에서 확인된 값, 단위: Ah)
Qmax = 56.168

# SOC 컬럼 자동 탐색
bms_columns = [col.lower() for col in bms_df.columns]
soc_col = next((col for col in bms_columns if 'soc' in col), None)

if soc_col:
    soc_series = bms_df[bms_df.columns[bms_columns.index(soc_col)]].dropna()
    soc_0 = soc_series.iloc[0] / 100  # 초기 SOC (0~1)
    soc_1 = soc_series.iloc[-1] / 100  # 최종 SOC (0~1)

    # E_stored 계산 함수
    def compute_estored(soc):
        result, _ = quad(ocv_func, 0, soc, limit=100)
        return Qmax * result  # 단위: Wh

    e_stored_0 = compute_estored(soc_0)
    e_stored_1 = compute_estored(soc_1)
    e_stored_diff = e_stored_0 - e_stored_1

    print(f"SOC_0: {soc_0*100:.1f}%, E_stored_0: {e_stored_0:.2f} Wh")
    print(f"SOC_1: {soc_1*100:.1f}%, E_stored_1: {e_stored_1:.2f} Wh")
    print(f"E_stored_0 - E_stored_1: {e_stored_diff:.2f} Wh")

else:
    print("SOC 컬럼을 찾을 수 없습니다.")

# 단위: 초당 측정, packcurrent 단위는 A라고 가정, 전압(V)도 있어야 Wh 계산 가능
# 필요한 컬럼 예시: 'chrg_cable_conn', 'speed', 'packcurrent', 'packvoltage'
# 정확한 이름은 bms_df.columns 로 확인 후 수정

df = bms_df.copy()

# 컬럼명 표준화 (소문자 처리)
df.columns = [col.lower() for col in df.columns]

# 컬럼명 매핑 (수동 확인 후 수정 가능)
cable_col = 'chrg_cable_conn'
speed_col = 'emobility_spd'
current_col = 'pack_current'
voltage_col = 'pack_volt'

# 시간 해상도 (초 단위 간격 가정, 변경 시 수정 필요)
time_interval = 1  # 초

# 에너지 계산 함수 (Wh): E = V * I * t / 3600
def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600  # Wh

# 조건별 에너지 계산
# 1. trip_discharge (주행 중 + 대기 소모)
cond_discharge_drive = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)
cond_discharge_idle = (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)
trip_discharge_energy = calc_energy(df[voltage_col], df[current_col], time_interval)
E_trip_discharge = trip_discharge_energy[cond_discharge_drive | cond_discharge_idle].sum()

# 2. trip_charge (회생제동)
cond_trip_charge = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] < 0)
E_trip_charge = calc_energy(df[voltage_col], df[current_col], time_interval)[cond_trip_charge].sum()

# 3. charging_energy (외부 충전)
cond_charging = (df[cable_col] == 1) & (df[current_col] < 0)
cond_charging_idle = (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)
charging_energy = calc_energy(df[voltage_col], df[current_col], time_interval)
E_charging = charging_energy[cond_charging].sum() - charging_energy[cond_charging_idle].sum()

# 4. trip_net = discharge - regen
E_trip_net = E_trip_discharge - abs(E_trip_charge)  # 회생은 에너지 회수라 절댓값 처리

# 출력
print(f"E_trip_discharge: {E_trip_discharge:.2f} Wh")
print(f"E_trip_charge (regen): {E_trip_charge:.2f} Wh")
print(f"E_trip_net: {E_trip_net:.2f} Wh")
print(f"E_charging: {E_charging:.2f} Wh")
