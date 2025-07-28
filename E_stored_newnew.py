import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\3week_data\bms_01241225206_2023-08.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"


bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], format='mixed', errors='coerce')
#bms_df['time'] = pd.to_datetime(bms_df['time']) # 데이터를 문자열에서 datetime 자료형으로 바꿔주어야, 시간순 정렬과 간격계산 가능.
#bms_df = bms_df.sort_values('time') # 시간순으로 정렬

bms_df = bms_df.sort_values('time')  # 시간 정렬

bms_df['delta_sec'] = bms_df['time'].diff().dt.total_seconds().fillna(0)



#bms_df['time']=bms_df['time'].str.strip()
#bms_df['time'] = pd.to_datetime(bms_df['time'], format='%Y-%m-%d %H:%M:%S')


#  2. OCV 곡선 로드 및 전처리
# ====================================
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')#엑셀파일을 열고, soc-ocv시트 읽기
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1 #엑셀시트에서 두번째 열을 검사하여 SOC(%)이름의 행을 찾음.
                                           #조건을 만족하는 첫번째 행 --> .index[0] 그다음 +1은 soc(%) 바로 다음행인 숫자데이터
                                             #행번호를 start row로 지정.
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]] # start_row: -> strat_row 행부터 끝까지.##0.05C_rate 채택
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
Qmax = 56.47*192*2

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

    soc_0 = 0.775#soc_series.iloc[0] / 100 #.iloc[0]으로 첫번째 행 선택.
    soc_1 = 0.785#soc_series.iloc[-1] / 100  #.iloc[-1] 로 마지막행 선택.

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

#df['time']=pd.to_datetime(df['time'])
#df['delta_sec']=df['time'].diff().dt.total_seconds().fillna(0)

#df['delta_sec'] = df['time'].diff().dt.total_seconds().fillna(0)
#(1).diff()를 이용하여 time열의 간격 계산 (2).dt.total_seconds()를 이용하여 시간간격을 초단위로 변환
# 3).fillna(0)을 이용해 비어있는 첫행을 0으로 채워주어 시간간격 계산


time_interval = 'delta_sec'


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

E_trip_discharge_drive=calc_energy(df[voltage_col], df[current_col],df[time_interval])[cond_discharge_drive].sum()
E_trip_discharge_idle=calc_energy(df[voltage_col], df[current_col],df[time_interval])[cond_discharge_idle].sum()
E_trip_discharge=E_trip_discharge_drive+E_trip_discharge_idle

E_trip_charge = abs(calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_trip_charge].sum())
#charging_energy = calc_energy(df[voltage_col], df[current_col], time_interval)

E_real_charging = abs(calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_charging].sum())
E_charging_idle = abs(calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_charging_idle].sum())

E_charging = E_real_charging - E_charging_idle
#E_charging = abs(charging_energy[cond_charging].sum() - charging_energy[cond_charging_idle].sum())

E_trip_net = E_trip_discharge - E_trip_charge






# ====================================
#  결과 출력
# ====================================

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
# 효율 계산 (조건적)
if (E_charging + e_stored_diff) > 0:
    efficiency1 = E_trip_net / (E_charging + e_stored_diff)*100
    print(f"Efficiency e1 = {efficiency1:.4f}%")
    efficiency2= (E_trip_net+e_stored_1)/(E_charging+e_stored_0)*100
    print(f"Efficiency e2 = {efficiency2:.4f}%")
    print(f".................................")

    #efficiency_charge= e_stored_diff/E_charging*100
    #print(f"Efficiency e_charge = {efficiency_charge:.4f}%")
    #efficiency_trip = E_trip_net/e_stored_diff *100
    #print(f"Efficiency e_trip = {efficiency_trip:.4f}%")

    print(bms_df['time'].head(5))
    print(bms_df['time'].head(10).tolist())


else:
    print("⚠️ Efficiency 계산 불가: 분모가 0 또는 음수")




#import matplotlib.pyplot as plt
#import matplotlib.font_manager as fm

# Windows 한글 폰트 예시 (맑은 고딕)
#plt.rcParams['font.family'] = 'Malgun Gothic'
#plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지
#import matplotlib.pyplot as plt
#plt.figure(figsize=(12,4))
#plt.plot(df['time'].iloc[1:], df['delta_sec'].iloc[1:], marker='o', linestyle='-')
#plt.title('시간 간격(delta_sec) 변화 추이')
#plt.xlabel('시간')
#plt.ylabel('초 단위 시간 간격')
#plt.grid(True)
#plt.show()

#large_gaps = df[df['delta_sec'] > 60][['time', 'delta_sec']]
#print(large_gaps)
#large_gaps.to_csv('large_gaps.csv', index=True)

# 예: 100개씩 나누어 출력
#for i in range(0, len(large_gaps), 100):
    #print(large_gaps.iloc[i:i+100])
    #input("계속 보려면 Enter 키를 누르세요...")


#plt.figure(figsize=(12,4))
#plt.plot(df.index[1:], df['delta_sec'].iloc[1:], marker='o', linestyle='-')
#plt.title('시간 간격(delta_sec) 변화 추이 (인덱스 기준)')
#plt.xlabel('데이터 인덱스')
#plt.ylabel('초 단위 시간 간격')
#plt.grid(True)
#plt.show()

