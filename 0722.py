import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

# ----------------------------
# 파일 로드 및 전처리
# ----------------------------
bms_df = pd.read_csv(r"C:\Users\OWNER\Desktop\EV6_2301\3week_data\bms_01241225206_2023-08.csv")
bms_df['time'] = pd.to_datetime(bms_df['time'], errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)

# segment 구분
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)
gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()
segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments
bms_df['delta_sec'] = bms_df.groupby('segment_id')['time'].diff().dt.total_seconds().fillna(0)
bms_df['valid'] = bms_df['delta_sec'] <= 20

# ----------------------------
# OCV 곡선 로딩
# ----------------------------
ocv_raw = pd.read_excel(r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx", sheet_name='SOC-OCV')
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

# ----------------------------
# 에너지 계산 함수
# ----------------------------
Qmax = 56.47
num_cells = 134

def compute_estored(soc):
    if soc <= 0.001:
        return 0
    try:
        result, _ = quad(ocv_func, 0, soc, limit=200)
        return Qmax * result * num_cells
    except Exception:
        return 0

# ----------------------------
# 가상 E_stored 계산 (전류 적분 기반)
# ----------------------------
def current_based_virtual_estored(group):
    Ah = (-group['pack_current'] * group['delta_sec'] / 3600).sum()
    delta_soc = Ah / (Qmax * num_cells)
    avg_soc = group['soc'].iloc[0] / 100 + delta_soc / 2
    e_start = compute_estored(avg_soc - delta_soc / 2)
    e_end = compute_estored(avg_soc + delta_soc / 2)
    return max(e_end - e_start, 0)

# ----------------------------
# 충전 구간 그룹핑 및 E_stored 계산
# ----------------------------
df = bms_df.copy()
df.columns = [col.lower() for col in df.columns]
df['chg_mask'] = (df['valid']) & (df['chrg_cable_conn'] == 1) & (df['pack_current'] < 0)
df['chg_segment'] = (df['chg_mask'] != df['chg_mask'].shift()).cumsum()
df.loc[~df['chg_mask'], 'chg_segment'] = np.nan
chg_groups = df[df['chg_segment'].notna()].groupby('chg_segment')

chg_e_stored_total = 0
chg_energy_total = 0

for seg_id, group in chg_groups:
    group = group.sort_values('time')
    soc_start = group['soc'].iloc[0] / 100
    soc_end = group['soc'].iloc[-1] / 100
    delta_soc = abs(soc_end - soc_start)

    if delta_soc < 0.001:
        e_stored = current_based_virtual_estored(group)
    else:
        e_stored = max(compute_estored(soc_end) - compute_estored(soc_start), 0)

    e_charging = (-group['pack_current'] * group['pack_volt'] * group['delta_sec'] / 3600).sum()

    chg_e_stored_total += e_stored
    chg_energy_total += e_charging

# ----------------------------
# 출력
# ----------------------------
if chg_energy_total > 0:
    eff = chg_e_stored_total / chg_energy_total * 100
    print(f"[🔋 충전 효율] e_chg = {eff:.2f}%")
    print(f" - 저장 에너지 총합: {chg_e_stored_total:.2f} Wh")
    print(f" - 실제 충전 에너지 총합: {chg_energy_total:.2f} Wh")
else:
    print("⚠️ 충전 구간 없음 또는 충전 에너지가 0")
