import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\0424raw\bms_01241124056_2024-04-24 (2).csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"


bms_df = pd.read_csv(bms_file)

# ====================================
#  2. OCV 곡선 로드 및 전처리
# ====================================
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')#엑셀파일을 열고, soc-ocv시트 읽기
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1 #엑셀시트에서 두번째 열을 검사하여 SOC(%)이름의 행을 찾음.
                                           #조건을 만족하는 첫번째 행 --> .index[0] 그다음 +1은 soc(%) 바로 다음행인 숫자데이터
                                             #행번호를 start row로 지정.
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]] # start_row: -> strat_row 행부터 끝까지.
                                                   # 그중 1=두번째열(soc), 2=세번째열(ocv(충전방향)

soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float) #.dropna() : 행단위로 NaN이 하나라도 있으면 그 행 제거. 전체가 제거됨.
                                                   # 동일표현 .dropna(axis=0, how='any')  NaN이 포함된 행 제거, NaN이
                                                                                         #하나라도 있으면 제거
                                                   #.astype(float) : df전체를 float(실수형)(자료형)으로 변환.
                                                   #적분이 가능한 수학연산이 되도록 변환
# OCV 보간 함수 정의 (SOC는 0~1로 스케일링)
soc_vals = soc_ocv_data['SOC'].values / 100    #soc를 0~1사이 값으로 변환
ocv_vals = soc_ocv_data['OCV'].values          #.values를 이용하여 series 타입을 Numpy배열로 변환. 인덱스 떼고 값만 쭉 나열.

# NaN 제거 및 정렬
valid_mask = (~np.isnan(soc_vals)) & (~np.isnan(ocv_vals)) #np.isnan함수. NaN인지 여부 확인 NaN이면 true반환.
                                                           # '~'를 통해서 NaN이면 false 반환. soc, ocv 모두 false
                                                           #  &(and)연산자를 통해 둘다 true일떄, 즉 둘다 NaN이 아닐때.
                                                           # 즉, NaN을 제외한 정상값들만 반환. True/False로 표시
soc_vals = soc_vals[valid_mask] # valid_mask를 통해서 NaN이 아닌값만 필터링/ 대괄호안에 Boolean배열. T/F배열 넣으면,
                                #True인 값들만 반환// soc_vals[valid_mask]는 valid mask가 True인 soc_vals만 선택
ocv_vals = ocv_vals[valid_mask] #마찬가지로 valid_mask를 통해 valid mask가 True인 ocv_vals만 반환

sort_idx = np.argsort(soc_vals) #soc_vals에 대하여 오름차순으로 정리한 "인덱스" 얻기. np.argsort()를 통해 오름차순 정렬시 인덱스 얻기
soc_vals = soc_vals[sort_idx] #얻어진 인덱스 sort_idx에 맞춰서 soc_vals를 오름차순으로 정렬
ocv_vals = ocv_vals[sort_idx] #마찬가지로 얻어진 인덱스에 맞춰서 오름차순 soc_vals에 대응하는 voc_vals를 정렬.
                                 #cf) 내림차순은 np.argsort()[::-1]

# 보간 함수
ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value="extrapolate") #(입력, 출력, 선형보간방법, soc값이 범위 벗어나면 외삽허용)
                                                                                 # 보간은 정해진 범위 내에서 추정, 외삽은 범위 밖에서 추정
                                                                                 #보간함수로 나타낸 ocv-soc커브
# ====================================
# 🔧 Qmax 값 설정 (Ah)
# ====================================
Qmax = 56.47*2*192

# ====================================
# 🔢 SOC 초기/최종값 추출
# ====================================

bms_columns = [col.lower() for col in bms_df.columns] #bms_df에 있는 columns(열)의 이름들을 col.lower()에 의해
                                                      #소문자로 변경하여 새롭게 bms_colulms로 만들어 저장.
                                                     #즉, Voltage, Current, SOC -> voltage, current, soc로 변환
soc_col = next((col for col in bms_columns if 'soc' in col), None) # bms_columns에 저장된 열이름 중에서 'soc'가 포함되어 있으면
                                                                   # 그 이름을 반환. ex) soc, soc(%)등을 반환. soc라는 글자가 없으면 None.
                                                                   #'soc'라는 글자가 포함된 첫번째 열을 soc_col로 저장.
                                                                   #next(..) 조건에 맞는 첫번쨰 값만 가져옴.
#soc_col=bms_df['soc']

if soc_col: #이전에 'soc'를 포함하는 첫번째 열에대하여 soc_col에 저장하였고 이것이 존재한다면 아래 계산 수행.
    soc_series = bms_df[bms_df.columns[bms_columns.index(soc_col)]].dropna() # 1st, bms_columns.index(soc_col) ..> soc_col에 해당하는
                                                                             # column이름이 bms_columns에서 몇번째인지 index 구함.
     #2nds, bms_df.columns[.index값.] -> 다시 소문자가 아닌 원래 이름으로 된 column명을 가져옴. ex) soc(%)의 인덱스 번호 찾고, 다시 SOC(%)컬럼명 가져옴.
     #3rd, bms_df[..SOC(%)..]에 해당하는 , SOC(%)에 해당하는 값들 가져옴.
     #4th, 가져온 값들에 대해 .dropna()를 통해 NaN(결측치)를 제거함.

    soc_0 = soc_series.iloc[-1] / 100 #.iloc[-1] 로 마지막행 선택.
    soc_1 = soc_series.iloc[0] / 100  #.iloc[0]으로 첫번째 행 선택.

    #에너지 계산 함수
    def compute_estored(soc):  #soc에 대하여 compute_estored 함수를 계산
        if soc <= 0.001:
            return 0
        try:
            result, _ = quad(ocv_func, 0, soc, limit=200) # 정적분함수 quad. ocv_func는 보간함수.
                                                  #result, _ 를 통해서 '_'의 의미는 오류값이 나오면 쓰지 않겠다는 뜻.
                                                   #limit=200 ; 구간을 200개로 나누어서 계산
            return Qmax * result  # Wh
        except Exception as e:  #적분을 계산하는 과정의 오류가 발생하면 에러메세지를 'e'라는 변수에 저장.
            print(f"[적분 실패] SOC: {soc:.4f} → 오류: {e}")
            return 0
    e_stored_0 = compute_estored(soc_0)
    e_stored_1 = compute_estored(soc_1)
    e_stored_diff = e_stored_0 - e_stored_1

    print(f"SOC_0: {soc_0 * 100:.1f}%, E_stored_0: {e_stored_0:.2f} Wh")
    print(f"SOC_1: {soc_1 * 100:.1f}%, E_stored_1: {e_stored_1:.2f} Wh")
    print(f"E_stored_0 - E_stored_1: {e_stored_diff:.2f} Wh")

else:
    print(" SOC 컬럼을 찾을 수 없습니다.")
    exit()

# ====================================

# E_trip_net, E_charging 계산
df = bms_df.copy()
df.columns = [col.lower() for col in df.columns]

cable_col = 'chrg_cable_conn'
speed_col = 'speed'
current_col = 'pack_current'
voltage_col = 'pack_volt'
time_interval = 2 #BMS데이터가 2초간격 데이터이므로.

def calc_energy(voltage, current, time_s): #E=V*I*time/3600 식을 사용하여 에너지 계산
    return (voltage * current * time_s) / 3600

# 조건 설정
cond_discharge_drive = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0) #(1) 주행만
cond_discharge_idle = (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0) #(3) 주차+충전기X+대기전력소모
cond_trip_charge = (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] < 0) #(2) 주행 중 break(회생제동)
cond_charging = (df[cable_col] == 1)  & (df[speed_col] == 0)&(df[current_col] < 0) #(4) 주차+충전기O
cond_charging_idle = (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0) #(5) 주차+충전기O+대기전력소모

# 에너지 계산
#trip_discharge_energy = calc_energy(df[voltage_col], df[current_col], time_interval)
#E_trip_discharge = trip_discharge_energy[cond_discharge_drive | cond_discharge_idle].sum()

E_trip_discharge_drive=calc_energy(df[voltage_col], df[current_col], time_interval)[cond_discharge_drive].sum()
E_trip_discharge_idle=calc_energy(df[voltage_col], df[current_col], time_interval)[cond_discharge_idle].sum()
E_trip_discharge=E_trip_discharge_drive+E_trip_discharge_idle

E_trip_charge = abs(calc_energy(df[voltage_col], df[current_col], time_interval)[cond_trip_charge].sum())
#charging_energy = calc_energy(df[voltage_col], df[current_col], time_interval)

E_real_charging = abs(calc_energy(df[voltage_col], df[current_col], time_interval)[cond_charging].sum())
E_charging_idle = abs(calc_energy(df[voltage_col], df[current_col], time_interval)[cond_charging_idle].sum())

E_charging = E_real_charging - E_charging_idle
#E_charging = abs(charging_energy[cond_charging].sum() - charging_energy[cond_charging_idle].sum())



E_trip_net = E_trip_discharge - E_trip_charge






# ====================================
# 📤 결과 출력
# ====================================

print(f"///////////////////////////////")
print(f"E_trip_discharge_drive: {E_trip_discharge_drive:.2f} Wh")
print(f"E_trip_discharge_idle: {E_trip_discharge_idle:.2f} Wh")
print(f"E_trip_discharge = E_trip_discharge_drive + E_trip_discharge_idle: {E_trip_discharge:.2f} Wh")
print(f"E_trip_charge (regen): {E_trip_charge:.2f} Wh")

print(f"E_trip_net = E_trip_discharge - E_trip_charge (regen) : {E_trip_net:.2f} Wh")
print(f"///////////////////////////////")
print(f"E_real_charging : {E_real_charging:.2f} Wh")
print(f"E_charging_idle : {E_charging_idle:.2f} Wh")

print(f"E_charging = E_real_charging - E_charging_idle: {E_charging:.2f} Wh")

print(f"///////////////////////////////")
# 효율 계산 (조건적)
if (E_charging + e_stored_diff) > 0:
    efficiency1 = E_trip_net / (E_charging + e_stored_diff)*100
    print(f"Efficiency e1 = {efficiency1:.4f}%")
    efficiency2= (E_trip_net+e_stored_1)/(E_charging+e_stored_0)*100
    print(f"Efficiency e2 = {efficiency2:.4f}%")
    print(f"///////////////////////////////")
    efficiency_charge= e_stored_diff/E_charging*100


    print(f"Efficiency e_charge = {efficiency_charge:.4f}%")
    efficiency_trip = E_trip_net/e_stored_diff *100
    print(f"Efficiency e_trip = {efficiency_trip:.4f}%")

else:
    print("⚠️ Efficiency 계산 불가: 분모가 0 또는 음수")





import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# ✅ 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux (예: Ubuntu)
    plt.rcParams['font.family'] = 'NanumGothic'  # 설치 필요: sudo apt install fonts-nanum

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 예: df['time'] 열이 datetime 형식이라고 가정
df['time'] = pd.to_datetime(df['time'])  # 혹시 문자열이면 datetime으로 변환

# SOC 시계열 준비
soc_series_full = df[soc_col].ffill() / 100
time_series = df['time']

# 회생제동 시점 인덱스
regen_indices = np.where(cond_trip_charge.values)[0]
regen_times = time_series.iloc[regen_indices]
regen_socs = soc_series_full.iloc[regen_indices]

# ✅ 시각화
plt.figure(figsize=(12,6))
plt.plot(time_series, soc_series_full, label='SOC 변화', color='black', linewidth=1)
plt.scatter(regen_times, regen_socs, color='limegreen', s=20, label='회생제동 시점', alpha=0.05)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # 시간 형식 설정
plt.xticks(rotation=45)  # 날짜 글자 기울이기

plt.title('SOC 변화')
plt.xlabel('시간')
plt.ylabel('SOC (0~1)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 엑셀의 실제 시간 열을 사용
time_series = pd.to_datetime(df['Time'])  # ← 'Time' 대신 실제 열 이름 사용
time_series = time_series[::-1]  # ← 시간 역순

# 나머지 부분은 동일
soc_series_full = df[soc_col].ffill() / 100
current_array = df[current_col].fillna(0).values
soc_array = soc_series_full.values

# 역순 처리
soc_array = soc_array[::-1]
current_array = current_array[::-1]


# SOC 및 전류
soc_series_full = df[soc_col].ffill() / 100
current_array = df[current_col].fillna(0).values
soc_array = soc_series_full.values

# 역순으로 바꾸기
soc_array = soc_array[::-1]
current_array = current_array[::-1]

# 📌 컬러 선 함수
def colorline(x, y, c, cmap='RdYlGn', linewidth=2):
    points = np.array([mdates.date2num(x), y]).T.reshape(-1, 1, 2)  # x축을 날짜 형식 숫자로 변환
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(c.min(), c.max())
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c)
    lc.set_linewidth(linewidth)
    return lc

# ✅ 시각화
fig, ax = plt.subplots(figsize=(12,6))
lc = colorline(time_series, soc_array, current_array)
ax.add_collection(lc)
ax.autoscale()

# x축 날짜 포맷 지정
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.xticks(rotation=45)

plt.title('SOC 변화 (전류값 기반 색상, 시간 역순)')
plt.xlabel('시간')
plt.ylabel('SOC (0~1)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.colorbar(lc, label='전류 (A)')
plt.tight_layout()
plt.show()







