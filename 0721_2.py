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
gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()

segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments

# 각 세그먼트에서 다시 delta_sec 계산
bms_df['delta_sec'] = (
    bms_df.groupby('segment_id')['time']
    .diff()
    .dt.total_seconds()
    .fillna(0)
)

# delta_sec > 10 제거용 마스크
bms_df['valid'] = bms_df['delta_sec'] <= 20

# ----------------------------
# OCV 곡선 처리
# ----------------------------
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 7]]
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
# SOC 기반 적분 함수
# ----------------------------
Qmax = 56.47

def compute_estored(soc):
    if soc <= 0.001:
        return 0
    try:
        result, _ = quad(ocv_func, 0, soc, limit=200)
        return Qmax * result * 134
    except Exception as e:
        print(f"[적분 실패] SOC: {soc:.4f} → 오류: {e}")
        return 0

# ----------------------------
# SOC 기반 전체 에너지 변화 계산
# ----------------------------
bms_columns = [col.lower() for col in bms_df.columns]
soc_col = next((col for col in bms_columns if 'soc' in col), None)

if soc_col:
    soc_series = bms_df.loc[bms_df['valid'], bms_df.columns[bms_columns.index(soc_col)]].dropna()
    soc_0 = soc_series.iloc[0] / 100
    soc_1 = soc_series.iloc[-1] / 100

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

def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600

cond_discharge_drive = df['valid'] & (df['chrg_cable_conn'] == 0) & (df['speed'] > 0) & (df['pack_current'] > 0)
cond_discharge_idle  = df['valid'] & (df['chrg_cable_conn'] == 0) & (df['speed'] == 0) & (df['pack_current'] > 0)
cond_trip_charge     = df['valid'] & (df['chrg_cable_conn'] == 0) & (df['speed'] > 0) & (df['pack_current'] < 0)
cond_charging        = df['valid'] & (df['chrg_cable_conn'] == 1) & (df['speed'] == 0) & (df['pack_current'] < 0)
cond_charging_idle   = df['valid'] & (df['chrg_cable_conn'] == 1) & (df['speed'] == 0) & (df['pack_current'] > 0)

E_trip_discharge_drive = calc_energy(df['pack_volt'], df['pack_current'], df['delta_sec'])[cond_discharge_drive].sum()
E_trip_discharge_idle  = calc_energy(df['pack_volt'], df['pack_current'], df['delta_sec'])[cond_discharge_idle].sum()
E_trip_discharge = E_trip_discharge_drive + E_trip_discharge_idle

E_trip_charge = calc_energy(df['pack_volt'], -df['pack_current'], df['delta_sec'])[cond_trip_charge].sum()
E_real_charging = calc_energy(df['pack_volt'], -df['pack_current'], df['delta_sec'])[cond_charging].sum()
E_charging_idle = calc_energy(df['pack_volt'], df['pack_current'], df['delta_sec'])[cond_charging_idle].sum()
E_charging = E_real_charging - E_charging_idle
E_trip_net = E_trip_discharge - E_trip_charge

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

# ----------------------------
# 보완된 저장 에너지 적분 방식 추가
# ----------------------------
def compute_segmentwise_estored(df, ocv_func, Qmax=56.47, n_cells=134):
    charging_mask = df['valid'] & (df['chrg_cable_conn'] == 1) & (df['pack_current'] < 0)
    df['chg_segment'] = (charging_mask != charging_mask.shift()).cumsum()
    df.loc[~charging_mask, 'chg_segment'] = np.nan

    chg_groups = df[df['chg_segment'].notna()].groupby('chg_segment')



chg_e_stored_total = 0
chg_energy_total = 0

SOC_FAKE_STEP_PER_SEC = 0.00001  # 전류에 비례한 SOC 증가 가중치

for seg_id, group in chg_groups:
    group = group.sort_values('time').copy()
    group['soc'] = group['soc'] / 100
    soc_vals = group['soc'].values
    curr_vals = group['pack_current'].values
    speed_vals = group['speed'].values
    conn_vals = group['chrg_cable_connected'].values
    delta_secs = group['delta_sec'].values

    soc_virtual = soc_vals[0]

    for i in range(1, len(soc_vals)):
        soc_prev = soc_virtual
        soc_curr = soc_vals[i]
        delta_soc_raw = soc_curr - soc_vals[i - 1]

        is_charging = (curr_vals[i] < -0.1) and (speed_vals[i] < 1) and (conn_vals[i] > 0)

        if delta_soc_raw > 0.0005:
            # 실제 SOC 증가 → 정상 적분
            e_start = compute_estored(soc_prev)
            e_end = compute_estored(soc_curr)
            chg_e_stored_total += max(e_end - e_start, 0)
            soc_virtual = soc_curr  # update
        elif is_charging:
            # SOC 변화 없지만 충전 조건 → 가상 SOC 적분
            fake_delta_soc = -curr_vals[i] * delta_secs[i] * SOC_FAKE_STEP_PER_SEC
            soc_next = soc_prev + fake_delta_soc

            # 적분 계산
            e_start = compute_estored(soc_prev)
            e_end = compute_estored(soc_next)
            chg_e_stored_total += max(e_end - e_start, 0)
            soc_virtual = soc_next  # 누적 SOC 업데이트

    # 실제 충전 에너지
    group['e_chg'] = abs(group['pack_volt'] * group['pack_current'] * group['delta_sec']) / 3600
    chg_energy_total += group['e_chg'].sum()

# 효율 계산
e_chg_eff = (chg_e_stored_total / chg_energy_total) * 100 if chg_energy_total > 0 else 0

print(f"[🔋 충전 효율] e_chg = {e_chg_eff:.2f}%")
print(f" - 저장 에너지 합: {chg_e_stored_total:.2f} Wh")
print(f" - 실제 충전 에너지 합: {chg_energy_total:.2f} Wh")






