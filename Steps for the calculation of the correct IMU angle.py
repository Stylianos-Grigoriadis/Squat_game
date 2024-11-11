import matplotlib.pyplot as plt
import pandas as pd
import lib_squats as lbsq
import math
from scipy.constants import g
import os

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
    df = df.iloc[:, :-1] # the last column have no use so we erase it
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

        angles = lbsq.q_to_ypr(q)
        yaw.append(angles[0])
        pitch.append(angles[1])
        roll.append(angles[2])
    df['Yaw'] = yaw
    df['Pitch'] = pitch
    df['Roll'] = roll

    return df

# path = r'path'
# os.chdir(path) # determine the specific path where the file is in

Squat = pd.read_csv(r'file_name.csv') # Read the file
Squat = convert_KIVNENT_IMU_to_readable_file(Squat) # Convert the file into readable data, and calculate the Yaw, Pitch, and Roll angles

plt.plot(Squat['Yaw'], label='Yaw')
plt.plot(Squat['Pitch'], label='Pitch')
plt.plot(Squat['Roll'], label='Roll')
plt.legend()
plt.show() # Show all angles

correct_angle_for_squat = Squat['Yaw'] # Determine the correct IMU angle for the movement of the avatar