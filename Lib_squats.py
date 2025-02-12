import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
from fathon import fathonUtils as fu
import fathon
import lib
from scipy.constants import g
import math


def DFA(variable):
    a = fu.toAggregated(variable)
        #b = fu.toAggregated(b)

    pydfa = fathon.DFA(a)

    winSizes = fu.linRangeByStep(start=4, end=int(len(variable)/4))
    revSeg = True
    polOrd = 1

    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    H, H_intercept = pydfa.fitFlucVec()
    # plt.plot(np.log(n), np.log(F), 'ro')
    # plt.plot(np.log(n), H_intercept + H * np.log(n), 'k-', label='H = {:.2f}'.format(H))
    # plt.xlabel('ln(n)', fontsize=14)
    # plt.ylabel('ln(F(n))', fontsize=14)
    # plt.title('DFA', fontsize=14)
    # plt.legend(loc=0, fontsize=14)
    # #plt.clf()
    # plt.show()
    return H

def ratio_0_to_100(data_series):
    """ Takes a data series and converts it into values from 0 to 100"""
    data_series = np.array(data_series)
    data_series = (data_series - np.min(data_series)) / (np.max(data_series) - np.min(data_series))*100

    return data_series

def ratio_0_to_1(data_series):
    """ Takes a data series and converts it into values from 0 to 1"""
    data_series = np.array(data_series)
    data_series = (data_series - np.min(data_series)) / (np.max(data_series) - np.min(data_series))

    return data_series

def pink_noise_x_y(N):
    """ This function creates a pink noise signal and then appends 1 value to the x_data and 1 value to the y_data"""
    pink_noise = cn.powerlaw_psd_gaussian(1,N*2)
    pink_noise = ratio_0_to_1(pink_noise)
    x_data = []
    y_data = []

    for i in range(1,len(pink_noise),2):
        y_data.append(pink_noise[i])
    for i in range(0,len(pink_noise),2):
        x_data.append(pink_noise[i])
    x_data = ratio_0_to_100(x_data)
    y_data = ratio_0_to_100(y_data)

    return x_data, y_data

def trigonometry(Orientation, TD):
    y_diff = np.sin(Orientation) * TD
    x_diff = np.cos(Orientation) * TD
    return x_diff, y_diff

def pink_noise_travel_distance_and_orientation(N):
    TD = cn.powerlaw_psd_gaussian(1, N)
    TD = ratio_0_to_1(TD)
    # Oriantation random
    Orientation = np.random.randn(1000)
    # Orientation pink
    # Orientation = cn.powerlaw_psd_gaussian(1, 1000)
    Orientation = ratio_0_to_1(Orientation)
    Orientation = Orientation * 359
    print(f'TD: {DFA(TD)}')
    print(f'Orientation: {DFA(Orientation)}')
    x_data = [0.5]
    y_data = [0.5]

    for i in range(len(TD)):
        if Orientation[i] <= 90:
            x_diff, y_diff = trigonometry(Orientation[i], TD[i])
            x_data.append(x_data[-1] + x_diff)
            y_data.append(y_data[-1] + y_diff)
        elif (Orientation[i] <= 180 and Orientation[i] > 90):
            x_diff, y_diff = trigonometry(Orientation[i], TD[i])
            x_data.append(x_data[-1] - x_diff)
            y_data.append(y_data[-1] + y_diff)
        elif (Orientation[i] <= 270 and Orientation[i] > 180):
            x_diff, y_diff = trigonometry(Orientation[i], TD[i])
            x_data.append(x_data[-1] - x_diff)
            y_data.append(y_data[-1] - y_diff)
        elif (Orientation[i] <= 360 and Orientation[i] > 270):
            x_diff, y_diff = trigonometry(Orientation[i], TD[i])
            x_data.append(x_data[-1] + x_diff)
            y_data.append(y_data[-1] - y_diff)

    x_data = ratio_0_to_100(x_data)
    y_data = ratio_0_to_100(y_data)

    return x_data, y_data

def pink_noise_and_derivative(N):
    pink_signal = cn.powerlaw_psd_gaussian(1,N)
    derivative = lib.derivative(pink_signal,1)
    derivative = list(derivative)
    derivative.append(derivative[-1])
    x_data = ratio_0_to_100(pink_signal)
    y_data = ratio_0_to_100(derivative)

    return x_data, y_data

def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return np.array([x_dot, y_dot, z_dot])

def lorenz_x_data_y_data(dt, N):
    xyzs = np.empty((N + 1, 3))
    xyzs[0] = (0., 1., 1.05)
    for i in range(N):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    x, y, z = xyzs.T
    x_data = ratio_0_to_100(x)
    y_data = ratio_0_to_100(y)

    return x_data, y_data

def aizawa(state, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = state
    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = c + a * z - (z ** 3) / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * x ** 3
    return np.array([dx, dy, dz])

def aizawa_x_data_y_data(dt, N):
    xyzs = np.empty((N + 1, 3))
    xyzs[0] = (0.1, 0, 0)

    for i in range(N):
        xyzs[i + 1] = xyzs[i] + aizawa(xyzs[i]) * dt

    x, y, z = xyzs.T
    x_data = ratio_0_to_100(x)
    y_data = ratio_0_to_100(y)

    return x_data, y_data

def erase_attractor_values(attractor, desired_number):
    attractor = list(attractor)
    cut_number = int(len(attractor)/desired_number)
    new_attractor = attractor[::cut_number]

    return new_attractor

def pink_noise_x_and_y(N):
    """ This function creates 2 different and seperate pink noise signals"""
    x_data = cn.powerlaw_psd_gaussian(1, N)
    y_data = cn.powerlaw_psd_gaussian(1, N)
    x_data = ratio_0_to_100(x_data)
    y_data = ratio_0_to_100(y_data)

    return x_data, y_data

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

    yaw = []
    pitch = []
    roll = []
    for i in range(len(df['Q_0'])):
        q = []
        q.append(df['Q_0'][i])
        q.append(df['Q_1'][i])
        q.append(df['Q_2'][i])
        q.append(df['Q_3'][i])

        angles = lib.q_to_ypr(q)
        yaw.append(angles[0])
        pitch.append(angles[1])
        roll.append(angles[2])
    df['Yaw'] = yaw
    df['Pitch'] = pitch
    df['Roll'] = roll

    return df
    
def values_during_game(df):
    """
    This function takes the whole data frame and returns only the rows where the target_pos_y are not None.
    This way we only take the rows in which we have targets"""
    df = converting_str_into_float(df, 'target_pos_y')

    filtered_data = df[df['target_pos_y'] != None]
    print('Hello')
    filtered_data = converting_str_into_float(filtered_data, 'target_pos_y')

    return filtered_data

def converting_str_into_float(df, column_name):
    df["FloatColumn"] = pd.to_numeric(df[column_name], errors="coerce")  # Invalid values become NaN
    return df
