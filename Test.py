import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
from fathon import fathonUtils as fu
import fathon
import lib_squats as lbsq
import math
from scipy.constants import g


path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Stylianos 11.5.2024'
squat = pd.read_csv(path+r'\Squat.csv')
on_table = pd.read_csv(path+r'\Stable on table.csv')
squat_0_degree = pd.read_csv(path+r'\Squat_0_degree.csv')
squat_180_degree = pd.read_csv(path+r'\Squat_180_degree.csv')

squat = lbsq.convert_KIVNENT_IMU_to_readable_file(squat)
on_table = lbsq.convert_KIVNENT_IMU_to_readable_file(on_table)
squat_0_degree = lbsq.convert_KIVNENT_IMU_to_readable_file(squat_0_degree)
squat_180_degree = lbsq.convert_KIVNENT_IMU_to_readable_file(squat_180_degree)

yaw = squat_180_degree['Yaw'].to_numpy()
pitch = squat_180_degree['Pitch'].to_numpy()
roll = squat_180_degree['Roll'].to_numpy()

yaw = yaw
pitch = pitch
roll = roll

print(squat)
print(on_table)
fig, axs = plt.subplots(2, 1)
axs[0].plot(squat['Yaw'], color='red', label='Yaw')
axs[0].plot(squat['Pitch'], color='blue', label='Pitch')
axs[0].plot(squat['Roll'], color='green', label='Roll')
axs[0].set_title('Squat')
axs[0].legend()


axs[1].plot(on_table['Yaw'], color='red', label='Yaw')
axs[1].plot(on_table['Pitch'], color='blue', label='Pitch')
axs[1].plot(on_table['Roll'], color='green', label='Roll')
axs[1].set_title('On table')
axs[1].legend()

plt.tight_layout()
plt.show()

plt.plot(squat['Yaw'], color='red', label='Yaw')
plt.plot(squat['Pitch'], color='blue', label='Pitch')
plt.plot(squat['Roll'], color='green', label='Roll')
plt.legend()
plt.title('Squat', fontdict={'fontsize': 16, 'fontweight': 'bold', 'family': 'serif', 'color': 'k'})
plt.show()


fig, axs = plt.subplots(3, 1)
axs[0].plot(squat['Yaw'], color='red', label='squat')
axs[0].plot(squat_0_degree['Yaw'], color='blue', label='squat_0_degree')
axs[0].plot(yaw, color='green', label='squat_180_degree')
axs[0].set_title('Yaw')
axs[0].legend()

axs[1].plot(squat['Pitch'], color='red', label='squat')
axs[1].plot(squat_0_degree['Pitch'], color='blue', label='squat_0_degree')
axs[1].plot(pitch, color='green', label='squat_180_degree')
axs[1].set_title('Pitch')
axs[1].legend()

axs[2].plot(squat['Roll'], color='red', label='squat')
axs[2].plot(squat_0_degree['Roll'], color='blue', label='squat_0_degree')
axs[2].plot(roll, color='green', label='squat_180_degree')
axs[2].set_title('Roll')
axs[2].legend()

plt.tight_layout()
plt.show()

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

def convert_magnetic_field_density(mag):
    """
    This function takes data points of Kinvent IMU angular velocity and returns angular velocity in the range of +-2000°/s
    """
    mag = (mag - 32768) * (2500.0/32768)
    return mag

def q_to_ypr(q):
    """ Import q as a list of quaternion"""
    if q:
        yaw = (math.atan2(2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[0] ** 2 + 2 * q[1] ** 2 - 1))
        roll = (-1 * math.asin(2 * q[1] * q[3] + 2 * q[0] * q[2]))
        pitch = (math.atan2(2 * q[2] * q[3] - 2 * q[0] * q[1], 2 * q[0] ** 2 + 2 * q[3] ** 2 - 1))
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        return [yaw, pitch, roll]
def convert_KIVNENT_IMU_to_readable_file(df):
    columns_IMU = ['Time', 'Q_0', 'Q_1', 'Q_2', 'Q_3', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Ang_Vel_X', 'Ang_Vel_Y', 'Ang_Vel_Z',
                   'Mag_X', 'Mag_Y', 'Mag_Z']
    df = df.iloc[:, :-1]
    df.columns = columns_IMU
    df['Q_0'] = df['Q_0'].apply(convert_quaternion)
    df['Q_1'] = df['Q_1'].apply(convert_quaternion)
    df['Q_2'] = df['Q_2'].apply(convert_quaternion)
    df['Q_3'] = df['Q_3'].apply(convert_quaternion)

    df['Acc_X'] = df['Acc_X'].apply(convert_acceleration)
    df['Acc_Y'] = df['Acc_Y'].apply(convert_acceleration)
    df['Acc_Z'] = df['Acc_Z'].apply(convert_acceleration)

    df['Ang_Vel_X'] = df['Ang_Vel_X'].apply(convert_ang_vel)
    df['Ang_Vel_Y'] = df['Ang_Vel_Y'].apply(convert_ang_vel)
    df['Ang_Vel_Z'] = df['Ang_Vel_Z'].apply(convert_ang_vel)

    df['Mag_X'] = df['Mag_X'].apply(convert_magnetic_field_density)
    df['Mag_Y'] = df['Mag_Y'].apply(convert_magnetic_field_density)
    df['Mag_Z'] = df['Mag_Z'].apply(convert_magnetic_field_density)
    print(df['Q_0'])
    print(df['Q_1'])
    print(df['Q_2'])
    print(df['Q_3'])

    yaw = []
    pitch = []
    roll = []
    for i in range(len(df['Q_0'])):
        q = []
        q.append(df['Q_0'][i])
        q.append(df['Q_1'][i])
        q.append(df['Q_2'][i])
        q.append(df['Q_3'][i])

        if i == 6:

            print(df['Q_0'][i])
            print(df['Q_1'][i])
            print(df['Q_2'][i])
            print(df['Q_3'][i])

        angles = lbsq.q_to_ypr(q)
        yaw.append(angles[0])
        pitch.append(angles[1])
        roll.append(angles[2])
    df['Yaw'] = yaw
    df['Pitch'] = pitch
    df['Roll'] = roll

    return df
path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Stylianos 11.11.2024\IMU on the table'
# path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Stylianos 11.5.2024'

Squat_back = pd.read_csv(path+r'\Squat button to back 2.csv')
Squat_back = convert_KIVNENT_IMU_to_readable_file(Squat_back)

Squat_front = pd.read_csv(path+r'\Squat button to front 2.csv')
Squat_front = convert_KIVNENT_IMU_to_readable_file(Squat_front)


plt.plot(Squat_back['Yaw'], label='Squat_back')
plt.plot(Squat_front['Yaw'], label='Squat_front')
plt.legend()
plt.show()
