

regen_groups = df[df['is_regen']].groupby('regen_group')
valid_regen_indices = []

# 시간 조건 필터링: 회생이 1초 이상 지속되는 구간만 인정
for group_id, group in regen_groups:
    duration = group['delta_sec'].sum()
    if duration >= 1.0:  # 1초 이상 지속되는 회생만 유효
        valid_regen_indices.extend(group.index)

df['is_valid_regen'] = df.index.isin(valid_regen_indices)

# 최종 회생 데이터셋
df_regen = df[df['is_valid_regen']].copy()
df_regen['energy_Wh'] = df_regen['pack_power_watt'] * df_regen['delta_sec'] / 360

# 방전 에너지 총합 (양수)
total_discharge_Wh = df_drive['energy_Wh'].sum()
# 회생 에너지 총합 (음수이므로 절댓값 취함)
total_regen_Wh = -df_regen['energy_Wh'].sum()  # 음수라서 부호 반전
# 순 소비 에너지 (방전 - 회생)
net_energy_Wh = total_discharge_Wh - total_regen_Wh
