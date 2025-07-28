import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

file_path = r"C:\Users\OWNER\Desktop\EV6_2301\bms_01241225206_2023-01 (1).csv"
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(df['time'], df['pack_current'], label='Pack Current (A)', color='b')
ax1.set_xlabel('Time')
ax1.set_ylabel('Pack Current (A)')
ax1.set_title('Pack Current vs. Time')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True)
ax1.legend()

ax2.plot(df['time'], df['pack_volt'], label='Pack Voltage (V)', color='r')
ax2.set_xlabel('Time')
ax2.set_ylabel('Pack Voltage (V)')
ax2.set_title('Pack Voltage vs. Time')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# 텍스트 객체를 미리 만들어 둠
text1 = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, verticalalignment='top')
text2 = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, verticalalignment='top')

def on_move(event):
    if event.inaxes == ax1 and event.xdata and event.ydata:
        time_val = pd.to_datetime(event.xdata, unit='D', origin='unix')
        text1.set_text(f'Time: {time_val.strftime("%Y-%m-%d %H:%M:%S")}\nValue: {event.ydata:.2f}')
        fig.canvas.draw_idle()
    elif event.inaxes == ax2 and event.xdata and event.ydata:
        time_val = pd.to_datetime(event.xdata, unit='D', origin='unix')
        text2.set_text(f'Time: {time_val.strftime("%Y-%m-%d %H:%M:%S")}\nValue: {event.ydata:.2f}')
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()
