# 전체 코드: 충전/방전 구간 분리 기반 효율 계산 및 delta_sec 이상치 처리 포함

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, IntegrationWarning
import warnings

# 🔇 IntegrationWarning 무시
warnings.filterwarnings("ignore", category=IntegrationWarning)

# 🔧 파일 경로 설정
bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\3week_data\bms_08_halfmonth.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"


# 🔄 BMS 데이터 로드 및 정리
bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], format='mixed', errors='coerce')
bms_df = bms_df.sort_values('time')
delta_sec = bms_df['time'].diff().dt.total_seconds().copy()
delta_sec.iloc[0] = 0
delta_sec = delta_sec.mask(delta_sec > 10, 0)
bms_df['delta_sec'] = delta_sec

# 🔄 OCV 곡선 처리
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 7]]
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)
soc_vals = soc_ocv_data['SOC'].values / 100
ocv_vals = soc_ocv_data['OCV'].values
valid_mask = (~np.isnan(soc_vals)) & (~np.isnan(ocv_vals))
soc_vals = soc_vals[valid_mask]
ocv_vals = ocv_vals[valid_mask]
sort_idx = np.argsort(soc_vals)
soc_vals = soc_vals[sort_idx]
ocv_vals = ocv_vals[sort_idx]
ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value="extrapolate")

# ⚙️ Qmax 설정
Qmax = 56.47*134

# 🔧 에너지 계산 함수들
def compute_estored(soc):
    if soc <= 0.001:
        return 0
    try:
        if soc < 0.01:
            return Qmax * ocv_func(soc) * soc
        result, _ = quad(ocv_func, 0, soc, limit=200, epsabs=1e-6, epsrel=1e-6)
        return Qmax * result
    except Exception as e:
        print(f"[적분 실패] SOC: {soc:.4f} → 오류: {e}")
        return 0

def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600

def split_segments(df, condition_col):
    df = df.copy()
    df['cond'] = condition_col.astype(int)
    df['cond_shift'] = df['cond'].shift(fill_value=0)
    df['segment'] = (df['cond'] != df['cond_shift']).cumsum()
    return df[df['cond'] == 1].groupby('segment')

def compute_estored_diff_by_segment(segment_df, soc_col):
    try:
        soc_start = segment_df[soc_col].iloc[0] / 100
        soc_end = segment_df[soc_col].iloc[-1] / 100
        return compute_estored(soc_end) - compute_estored(soc_start)
    except:
        return 0

# 🔎 SOC 컬럼 찾기
bms_df.columns = [col.lower() for col in bms_df.columns]
soc_col = next((col for col in bms_df.columns if 'soc' in col), None)
if not soc_col:
    print("SOC 컬럼을 찾을 수 없습니다.")
    exit()

# 🔧 조건 정의
cable_col = 'chrg_cable_conn'
speed_col = 'speed'
current_col = 'pack_current'
voltage_col = 'pack_volt'

df = bms_df.copy()
cond_discharge_drive = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)
cond_discharge_idle = (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)
cond_trip_charge = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] < 0)
cond_charging = (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)
cond_charging_idle = (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)

# 🔋 E_trip, E_charging 계산
E_trip_discharge_drive = calc_energy(df[voltage_col], df[current_col], df['delta_sec'])[cond_discharge_drive].sum()
E_trip_discharge_idle = calc_energy(df[voltage_col], df[current_col], df['delta_sec'])[cond_discharge_idle].sum()
E_trip_discharge = E_trip_discharge_drive + E_trip_discharge_idle
E_trip_charge = abs(calc_energy(df[voltage_col], df[current_col], df['delta_sec'])[cond_trip_charge].sum())
E_trip_net = E_trip_discharge - E_trip_charge

E_real_charging = abs(calc_energy(df[voltage_col], df[current_col], df['delta_sec'])[cond_charging].sum())
E_charging_idle = abs(calc_energy(df[voltage_col], df[current_col], df['delta_sec'])[cond_charging_idle].sum())
E_charging = E_real_charging - E_charging_idle

# ⚡ e_charge 계산 (구간별)
charge_segments = split_segments(df, cond_charging)
chg_e_list = [compute_estored_diff_by_segment(g, soc_col) for _, g in charge_segments]
total_stored_gain = sum(chg_e_list)
e_charge_eff = total_stored_gain / E_real_charging if E_real_charging > 0 else 0

# ⚡ e_trip 계산 (구간별)
discharge_segments = split_segments(df, cond_discharge_drive | cond_discharge_idle)
dis_e_list = [compute_estored_diff_by_segment(g, soc_col) for _, g in discharge_segments]
total_stored_loss = -sum(dis_e_list)
e_trip_eff = E_trip_net / total_stored_loss if total_stored_loss > 0 else 0



# 🖨️ 출력
print("===============================")
print(f"E_trip_discharge_drive: {E_trip_discharge_drive:.2f}Wh")
print(f"E_trip_discharge_idle: {E_trip_discharge_idle:.2f}Wh")
print(f"E_trip_charge (regen): {E_trip_charge:.2f}Wh")
print(f"E_trip_net: {E_trip_net:.2f}Wh")
print("===============================")
print(f"E_real_charging: {E_real_charging:.2f}Wh")
print(f"E_charging_idle: {E_charging_idle:.2f}Wh")
print(f"E_charging: {E_charging:.2f}Wh")
print("===============================")
print(f"[e_chg] Stored Gain: {total_stored_gain:.2f} / Charging: {E_real_charging:.2f} → e_chg: {e_charge_eff*100:.2f}%")
print(f"[e_trip] Trip Net: {E_trip_net:.2f} / Stored Loss: {total_stored_loss:.2f} → e_trip: {e_trip_eff*100:.2f}%")
print("===============================")


