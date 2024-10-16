import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lib
from scipy.constants import g


def normalize_quaternion(q):
    norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    return [q[i] / norm for i in range(4)]

def convert_force_values(data_point):
    """
    This function takes data points of Kinvent force and returns force in kg
    """
    data_point = data_point/10
    return data_point

def convert_quaternion(quaternion):
    """
    This function takes data points of Kinvent IMU quaternions and returns quaternion from -1 to 1
    """
    quaternion = (quaternion - 32768)/16384
    return quaternion

def convert_acceleration(acc):
    """
    This function takes data points of Kinvent force and returns force in m/(s^2)
    """
    acc = (acc - 32768) * (16/32768)
    acc = acc * g
    return acc

def convert_ang_vel(vel):
    """
    This function takes data points of Kinvent IMU angular velocity and returns angular velocity in the range of +-2000°/s
    """
    vel = (vel - 32768) * (2000.0/32768)
    return vel

IMU = pd.read_csv(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Stylianos 10.14.2024\KFORCESens21158_EC_96_4B_1F_5C_FE.csv', header=None, delimiter=',')
ForcePlateR = pd.read_csv(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Stylianos 10.14.2024\KFORCEPlateR04801_FC_BF_8C_3B_14_04.csv', header=None, delimiter=',')
ForcePlateL = pd.read_csv(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Stylianos 10.14.2024\KFORCEPlateL04758_D2_80_A2_94_CB_1F.csv', header=None, delimiter=',')

IMU = IMU.iloc[:, :-1]
ForcePlateR = ForcePlateR.iloc[:, :-1]
ForcePlateL = ForcePlateL.iloc[:, :-1]

columns_IMU = ['Time', 'Q_0', 'Q_1', 'Q_2', 'Q_3', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Ang_Vel_X', 'Ang_Vel_Y', 'Ang_Vel_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
columns_ForcePlateR = ['Time', 'Sensor_1_R', 'Sensor_2_R', 'Sensor_3_R', 'Sensor_4_R']
columns_ForcePlateL = ['Time', 'Sensor_1_L', 'Sensor_2_L', 'Sensor_3_L', 'Sensor_4_L']

IMU.columns = columns_IMU
ForcePlateR.columns = columns_ForcePlateR
ForcePlateL.columns = columns_ForcePlateL

IMU['Q_0'] = IMU['Q_0'].apply(convert_quaternion)
IMU['Q_1'] = IMU['Q_1'].apply(convert_quaternion)
IMU['Q_2'] = IMU['Q_2'].apply(convert_quaternion)
IMU['Q_3'] = IMU['Q_3'].apply(convert_quaternion)

ForcePlateR['Sensor_1_R'] = ForcePlateR['Sensor_1_R'].apply(convert_force_values)
ForcePlateR['Sensor_2_R'] = ForcePlateR['Sensor_2_R'].apply(convert_force_values)
ForcePlateR['Sensor_3_R'] = ForcePlateR['Sensor_3_R'].apply(convert_force_values)
ForcePlateR['Sensor_4_R'] = ForcePlateR['Sensor_4_R'].apply(convert_force_values)
ForcePlateL['Sensor_1_L'] = ForcePlateL['Sensor_1_L'].apply(convert_force_values)
ForcePlateL['Sensor_2_L'] = ForcePlateL['Sensor_2_L'].apply(convert_force_values)
ForcePlateL['Sensor_3_L'] = ForcePlateL['Sensor_3_L'].apply(convert_force_values)
ForcePlateL['Sensor_4_L'] = ForcePlateL['Sensor_4_L'].apply(convert_force_values)

IMU['Acc_X'] = IMU['Acc_X'].apply(convert_acceleration)
IMU['Acc_Y'] = IMU['Acc_Y'].apply(convert_acceleration)
IMU['Acc_Z'] = IMU['Acc_Z'].apply(convert_acceleration)

IMU['Ang_Vel_X'] = IMU['Ang_Vel_X'].apply(convert_ang_vel)
IMU['Ang_Vel_Y'] = IMU['Ang_Vel_Y'].apply(convert_ang_vel)
IMU['Ang_Vel_Z'] = IMU['Ang_Vel_Z'].apply(convert_ang_vel)


ForcePlateR = ForcePlateR.iloc[:, 1:]
ForcePlateL = ForcePlateL.iloc[:, 1:]

print(IMU)

all_data = pd.concat([IMU, ForcePlateR], axis=1)
all_data = pd.concat([all_data, ForcePlateL], axis=1)
Total_force_R = (ForcePlateR['Sensor_1_R'] + ForcePlateR['Sensor_2_R'] + ForcePlateR['Sensor_3_R'] + ForcePlateR['Sensor_4_R']).tolist()
Total_force_L = (ForcePlateL['Sensor_1_L'] + ForcePlateL['Sensor_2_L'] + ForcePlateL['Sensor_3_L'] + ForcePlateL['Sensor_4_L']).tolist()

all_data['Total_force_R'] = Total_force_R
all_data['Total_force_L'] = Total_force_L

Total_force = [a + b for a, b in zip(Total_force_R, Total_force_L)]
all_data['Total_force'] = Total_force

print(all_data)

yaw = []
pitch = []
roll = []
for i in range(len(IMU['Q_0'])):
    q = []
    q.append(IMU['Q_0'][i])
    q.append(IMU['Q_1'][i])
    q.append(IMU['Q_2'][i])
    q.append(IMU['Q_3'][i])

    angles = lib.q_to_ypr(q)
    yaw.append(angles[0])
    pitch.append(angles[1])
    roll.append(angles[2])

# plt.plot(yaw, label='yaw')
# plt.plot(pitch, label='pitch')
# plt.plot(roll, label='roll')
# plt.legend()
# plt.show()

# plt.plot(IMU['Acc_X'], label='Acc_X')
# plt.plot(IMU['Acc_Y'], label='Acc_Y')
# plt.plot(IMU['Acc_Z'], label='Acc_Z')
# plt.legend()
# plt.show()

# plt.plot(IMU['Ang_Vel_X'], label='Ang_Vel_X')
# plt.plot(IMU['Ang_Vel_Y'], label='Ang_Vel_Y')
# plt.plot(IMU['Ang_Vel_Z'], label='Ang_Vel_Z')
# plt.legend()
# plt.show()

Ang_X_1 = lib.intergral(IMU['Ang_Vel_X'], 0.005)
Ang_Y_1 = lib.intergral(IMU['Ang_Vel_Y'], 0.005)
Ang_Z_1 = lib.intergral(IMU['Ang_Vel_Z'], 0.005)

# plt.plot(Ang_X_1, label='Ang_X_1')
# plt.plot(Ang_Y_1, label='Ang_Y_1')
# plt.plot(Ang_Z_1, label='Ang_Z_1')
# plt.legend()
# plt.show()

Ang_X_2 = []
Ang_Y_2 = []
Ang_Z_2 = []
for i in range(len(IMU['Time'])):
    Ang_X_2.append(0.005 * IMU['Ang_Vel_X'][i])
    Ang_Y_2.append(0.005 * IMU['Ang_Vel_Y'][i])
    Ang_Z_2.append(0.005 * IMU['Ang_Vel_Z'][i])

# plt.plot(Ang_X_2, label='Ang_X_2')
# plt.plot(Ang_Y_2, label='Ang_Y_2')
# plt.plot(Ang_Z_2, label='Ang_Z_2')
# plt.legend()
# plt.show()

fig, ax1 = plt.subplots()

ax1.plot(pitch, 'g-', label='pitch')  # Plot on the first y-axis (left)
ax1.set_ylabel('pitch', color='g')
ax1.set_ylim(-180,-70)

ax1.tick_params(axis='y', labelcolor='g')

# Create the second plot with the same x-axis but a different y-axis
ax2 = ax1.twinx()
ax2.plot(all_data['Total_force'], 'b-', label='Total_force')  # Plot on the second y-axis (right)
ax2.set_ylabel('Ground reaction force', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Show the plot
plt.title('Two plots with different y-axes')
plt.show()

