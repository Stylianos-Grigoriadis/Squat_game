import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lib_squats as lbsq

def force_np_array(df, column_name):
    array = df[column_name].to_numpy()
    return array

def force_both_plates(df):
    l_ch1 = force_np_array(df, 'l_ch1')
    l_ch2 = force_np_array(df, 'l_ch2')
    l_ch3 = force_np_array(df, 'l_ch3')
    l_ch4 = force_np_array(df, 'l_ch4')
    r_ch1 = force_np_array(df, 'r_ch1')
    r_ch2 = force_np_array(df, 'r_ch2')
    r_ch3 = force_np_array(df, 'r_ch3')
    r_ch4 = force_np_array(df, 'r_ch4')

    r_ch_all = r_ch1 + r_ch2 + r_ch3 + r_ch4
    l_ch_all = l_ch1 + l_ch2 + l_ch3 + l_ch4

    ch_all = r_ch_all + l_ch_all

    return ch_all

def quaternions_to_angles(df):
    yaw = []
    pitch = []
    roll = []
    for i in range(len(df['quat_w'])):
        q = []
        q.append(df['quat_w'][i])
        q.append(df['quat_x'][i])
        q.append(df['quat_y'][i])
        q.append(df['quat_z'][i])

        angles = lbsq.q_to_ypr(q)
        yaw.append(angles[0])
        pitch.append(angles[1])
        roll.append(angles[2])
    return yaw, pitch, roll

path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Thijs 10.29.2024'
list_headers = ["ts", "event", "l_ch1", "l_ch2", "l_ch3", "l_ch4", "r_ch1", "r_ch2", "r_ch3", "r_ch4",
           "quat_w", "quat_x", "quat_y", "quat_z", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y",
           "gyro_z", "magn_x", "magn_y", "magn_z"]

deep_squat = pd.read_csv(path + r'\deep_squad-10182024_123102.txt', skiprows=2, header=None, names=list_headers)
semi_squat = pd.read_csv(path + r'\semisquad-10182024_122713.txt', skiprows=2, header=None, names=list_headers)
squat = pd.read_csv(path + r'\squad-10182024_122811.txt', skiprows=2, header=None, names=list_headers)

for i in range(len(deep_squat['ts'])-1):
    print(deep_squat['ts'][i+1] - deep_squat['ts'][i])


# pd.set_option('display.max_rows', None)    # Show all rows
# pd.set_option('display.max_columns', None) # Show all columns

deep_squat_force = force_both_plates(deep_squat)
semi_squat_force = force_both_plates(semi_squat)
squat_force = force_both_plates(squat)

plt.plot(deep_squat_force, label='deep_squat_force')
plt.plot(semi_squat_force, label='semi_squat_force')
plt.plot(squat_force, label='squat_force')
plt.legend()
plt.show()

deep_squat_yaw, deep_squat_pitch, deep_squat_roll = quaternions_to_angles(deep_squat)
semi_squat_yaw, semi_squat_pitch, semi_squat_roll = quaternions_to_angles(semi_squat)
squat_yaw, squat_pitch, squat_roll = quaternions_to_angles(squat)

fig, axs = plt.subplots(3, 1)
axs[0].plot(deep_squat_yaw, label='deep_squat')
axs[0].plot(semi_squat_yaw, label='semi_squat')
axs[0].plot(squat_yaw, label='squat')
axs[0].set_title('Yaw')

axs[1].plot(deep_squat_pitch)
axs[1].plot(semi_squat_pitch)
axs[1].plot(squat_pitch)
axs[1].set_title('Pitch')

axs[2].plot(deep_squat_roll)
axs[2].plot(semi_squat_roll)
axs[2].plot(squat_roll)
axs[2].set_title('Roll')

axs[0].legend()

plt.show()



fig, axs = plt.subplots(3, 1)
fig.suptitle('Deep Squat and angles', fontsize=16)


axs[0].plot(deep_squat_force, color='blue')
axs[0].set_title('Force (blue) and Yaw (red)')
ax2_0 = axs[0].twinx()
ax2_0.plot(deep_squat_yaw, color='red')


axs[1].plot(deep_squat_force, color='blue')
axs[1].set_title('Force (blue) and Pitch (red)')
ax2_0 = axs[1].twinx()
ax2_0.plot(deep_squat_pitch, color='red')

axs[2].plot(deep_squat_force, color='blue')
axs[2].set_title('Force (blue) and Roll (red)')
ax2_0 = axs[2].twinx()
ax2_0.plot(deep_squat_roll, color='red')

plt.show()

fig, axs = plt.subplots(3, 1)
fig.suptitle('Semi Squat and angles', fontsize=16)


axs[0].plot(semi_squat_force, color='blue')
axs[0].set_title('Force (blue) and Yaw (red)')
ax2_0 = axs[0].twinx()
ax2_0.plot(semi_squat_yaw, color='red')


axs[1].plot(semi_squat_force, color='blue')
axs[1].set_title('Force (blue) and Pitch (red)')
ax2_0 = axs[1].twinx()
ax2_0.plot(semi_squat_pitch, color='red')

axs[2].plot(semi_squat_force, color='blue')
axs[2].set_title('Force (blue) and Roll (red)')
ax2_0 = axs[2].twinx()
ax2_0.plot(semi_squat_roll, color='red')

plt.show()

fig, axs = plt.subplots(3, 1)
fig.suptitle('Squat and angles', fontsize=16)


axs[0].plot(squat_force, color='blue')
axs[0].set_title('Force (blue) and Yaw (red)')
ax2_0 = axs[0].twinx()
ax2_0.plot(squat_yaw, color='red')


axs[1].plot(squat_force, color='blue')
axs[1].set_title('Force (blue) and Pitch (red)')
ax2_0 = axs[1].twinx()
ax2_0.plot(squat_pitch, color='red')

axs[2].plot(squat_force, color='blue')
axs[2].set_title('Force (blue) and Roll (red)')
ax2_0 = axs[2].twinx()
ax2_0.plot(squat_roll, color='red')

plt.show()

fig, axs = plt.subplots(3, 1)
fig.suptitle('Squat and angles', fontsize=16)


axs[0].plot(deep_squat_force, color='blue')
axs[0].set_title('Deep Squat Force (blue), Yaw (red), Pitch (orange), Roll (darkred)')
ax2_0 = axs[0].twinx()
ax2_0.plot(deep_squat_yaw, color='red')
ax2_0.plot(deep_squat_pitch, color='orange')
ax2_0.plot(deep_squat_roll, color='darkred')


axs[1].plot(semi_squat_force, color='blue')
axs[1].set_title('Semi Squat Force (blue), Yaw (red), Pitch (orange), Roll (darkred)')
ax2_1 = axs[1].twinx()
ax2_1.plot(semi_squat_yaw, color='red')
ax2_1.plot(semi_squat_pitch, color='orange')
ax2_1.plot(semi_squat_roll, color='darkred')

axs[2].plot(squat_force, color='blue')
axs[2].set_title('Squat Force (blue), Yaw (red), Pitch (orange), Roll (darkred)')
ax2_2 = axs[2].twinx()
ax2_2.plot(squat_yaw, color='red')
ax2_2.plot(squat_pitch, color='orange')
ax2_2.plot(squat_roll, color='darkred')

plt.show()





