import pandas as pd
import matplotlib.pyplot as plt

file_path=r"C:\Users\OWNER\Desktop\EV6_2301\bms_01241228049_2023-12-02.csv"
df=pd.read_csv(file_path)

df['time']=pd.to_datetime(df['time'])
df['delta_sec']=df['time'].diff().dt.total_seconds().fillna(0)


df['speed']=df['emobility_spd']
df['pack_power_watt']=df['pack_current']*df['pack_volt']


df=df.sort.values('time')


# 에너지 계산
E_trip_discharge_drive=calc_energy(df[voltage_col], df[current_col], time_interval)[cond_discharge_drive].sum() #(1) 주행만
E_trip_discharge_idle=calc_energy(df[voltage_col], df[current_col], time_interval)[cond_discharge_idle].sum() #(3) 주차+충전기X+대기전력소모
E_trip_discharge=E_trip_discharge_drive+E_trip_discharge_idle

E_trip_charge = abs(calc_energy(df[voltage_col], df[current_col], time_interval)[cond_trip_charge].sum()) #(2) 주행 중 break(회생제동)

E_trip_net = E_trip_discharge - E_trip_charge # 주행 중 순 소모된 에너지

E_real_charging = abs(calc_energy(df[voltage_col], df[current_col], time_interval)[cond_charging].sum()) #(4) 주차+충전기O
E_charging_idle = abs(calc_energy(df[voltage_col], df[current_col], time_interval)[cond_charging_idle].sum()) #(5) 주차+충전기O+대기전력소모
E_charging = E_real_charging - E_charging_idle # 순 충전된 에너지



