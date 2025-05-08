import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from data_collector_w_imu_visual_native import quat_to_ypr

emg_file = "examples/data/subj_0_fist_96_emg_rec_l.p"
imu_file = "examples/data/subj_0_fist_96_imu_rec_l.p"

pad = pickle.load(open(emg_file, "rb"))
imu = pickle.load(open(imu_file, "rb"))

new_emg = []
is_recording = []
emg_ticks = []
for sample in pad:
    new_emg.append(sample[0])
    is_recording.append(sample[1])
    emg_ticks.append(sample[2])

df_emg = pd.DataFrame(new_emg, columns=range(1, 9))

max = -10000000000
min = 10000000000
for emg_8_channel in new_emg:
    for each_channel in emg_8_channel:
        if max < each_channel:
            max = each_channel
        if min > each_channel:
            min = each_channel

save_emg_ticks = emg_ticks
emg_ticks = [
    (tick - emg_ticks[0]) / 1000 for tick in emg_ticks
]  # normalize emg_ticks and change milisec into sec


fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=11)

c = ["green" if a else "black" for a in is_recording]
subplot_list_emg = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
channel = 1
for i in subplot_list_emg:
    lines = [
        ((x0, y0), (x1, y1))
        for x0, y0, x1, y1 in zip(
            emg_ticks[:-1], df_emg[channel][:-1], emg_ticks[1:], df_emg[channel][1:]
        )
    ]
    colored_lines = LineCollection(lines, colors=c, linewidths=(1,))
    i.add_collection(colored_lines)
    i.autoscale_view()
    channel += 1

new_ypr = []
imu_ticks = []
for i in imu:
    [w, nx, ny, nz] = [x for x in i[0][0]]
    print(w, nx, ny, nz)
    [yaw, pitch, roll] = quat_to_ypr([w, nx, ny, nz])
    new_ypr.append([yaw, pitch, roll])
    imu_ticks.append(i[-1])
imu_ticks = [(tick - imu_ticks[0]) / 1000 for tick in imu_ticks]
df_imu = pd.DataFrame(new_ypr, columns=range(0, 3))

subplot_list_imu = [ax8, ax9, ax10]
legend = ["yaw", "pitch", "row"]
c = ["blue" if a else "black" for a in is_recording]
channel = 0
for i in subplot_list_imu:
    lines = [
        ((x0, y0), (x1, y1))
        for x0, y0, x1, y1 in zip(
            imu_ticks[:-1], df_imu[channel][:-1], imu_ticks[1:], df_imu[channel][1:]
        )
    ]
    colored_lines = LineCollection(lines, colors=c, linewidths=(1,))
    i.add_collection(colored_lines)
    i.autoscale_view()
    i.legend(title=legend[channel])
    channel += 1

plt.setp((ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7), ylim=(min, max))
plt.setp((ax8, ax9, ax10), ylim=(-180, 180))

# save to csv
save_df_emg = df_emg
save_df_emg.insert(8, "emg_ticks", save_emg_ticks)
save_df_emg.insert(9, "is_recording", is_recording)
save_df_imu = pd.DataFrame(
    imu, columns=["raw_imu", "adjusted_ypr", "neutral_ypr", "is_recording", "emg_ticks"]
)
save_df_emg.to_csv(emg_file[:-2] + ".csv")
save_df_imu.to_csv(imu_file[:-2] + ".csv")


plt.show()
