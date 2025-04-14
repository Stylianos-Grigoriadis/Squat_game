import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import colorednoise as cn
import lib
from scipy.constants import g
import math
from scipy import stats
from scipy.stats import pearsonr
import itertools
import time
from scipy.stats import linregress
import pwlf
from scipy.optimize import curve_fit



def DFA_NONAN(data, scales, order=1, plot=True):
    """Perform Detrended Fluctuation Analysis on data

    Inputs:
        data: 1D numpy array of time series to be analyzed.
        scales: List or array of scales to calculate fluctuations
        order: Integer of polynomial fit (default=1 for linear)
        plot: Return loglog plot (default=True to return plot)

    Outputs:
        scales: The scales that were entered as input
        fluctuations: Variability measured at each scale with RMS
        alpha value: Value quantifying the relationship between the scales
                     and fluctuations

....References:
........Damouras, S., Chang, M. D., Sejdi, E., & Chau, T. (2010). An empirical
..........examination of detrended fluctuation analysis for gait data. Gait &
..........posture, 31(3), 336-340.
........Mirzayof, D., & Ashkenazy, Y. (2010). Preservation of long range
..........temporal correlations under extreme random dilution. Physica A:
..........Statistical Mechanics and its Applications, 389(24), 5573-5580.
........Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
..........Quantification of scaling exponents and crossover phenomena in
..........nonstationary heartbeat time series. Chaos: An Interdisciplinary
..........Journal of Nonlinear Science, 5(1), 82-87.
# =============================================================================
                            ------ EXAMPLE ------

      - Generate random data
      data = np.random.randn(5000)

      - Create a vector of the scales you want to use
      scales = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]

      - Set a detrending order. Use 1 for a linear detrend.
      order = 1

      - run dfa function
      s, f, a = dfa(data, scales, order, plot=True)
# =============================================================================
"""

    # Check if data is a column vector (2D array with one column)
    if data.shape[0] == 1:
        # Reshape the data to be a column vector
        data = data.reshape(-1, 1)
    else:
        # Data is already a column vector
        data = data

    # =============================================================================
    ##########################   START DFA CALCULATION   ##########################
    # =============================================================================

    # Step 1: Integrate the data
    integrated_data = np.cumsum(data - np.mean(data))

    fluctuation = []

    for scale in scales:
        # Step 2: Divide data into non-overlapping window of size 'scale'
        chunks = len(data) // scale
        ms = 0.0

        for i in range(chunks):
            this_chunk = integrated_data[i * scale:(i + 1) * scale]
            x = np.arange(len(this_chunk))

            # Step 3: Fit polynomial (default is linear, i.e., order=1)
            coeffs = np.polyfit(x, this_chunk, order)
            fit = np.polyval(coeffs, x)

            # Detrend and calculate RMS for the current window
            ms += np.mean((this_chunk - fit) ** 2)

            # Calculate average RMS for this scale
        fluctuation.append(np.sqrt(ms / chunks))

        # Perform linear regression
    alpha, intercept = np.polyfit(np.log(scales), np.log(fluctuation), 1)

    # Create a log-log plot to visualize the results
    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(scales, fluctuation, marker='o', markerfacecolor='red', markersize=8,
                   linestyle='-', color='black', linewidth=1.7, label=f'Alpha = {alpha:.3f}')
        plt.xlabel('Scale (log)')
        plt.ylabel('Fluctuation (log)')
        plt.legend()
        plt.title('Detrended Fluctuation Analysis')
        plt.grid(True)
        plt.show()

    # Return the scales used, fluctuation functions and the alpha value
    return scales, fluctuation, alpha


def ratio_0_to_100(data_series):
    """ Takes a data series and converts it into values from 0 to 100"""
    data_series = np.array(data_series)
    data_series = (data_series - np.min(data_series)) / (np.max(data_series) - np.min(data_series)) * 100

    return data_series


def ratio_0_to_1(data_series):
    """ Takes a data series and converts it into values from 0 to 1"""
    data_series = np.array(data_series)
    data_series = (data_series - np.min(data_series)) / (np.max(data_series) - np.min(data_series))

    return data_series


def creation_rigid_signal(N):
    x_sequence = [50, 100, 50, 0]
    y_sequence = [100, 50, 0, 50]
    x_data = np.tile(x_sequence, N // len(x_sequence))
    y_data = np.tile(y_sequence, N // len(y_sequence))
    more = N - len(x_data)
    for i in range(0, more):
        x_data = np.append(x_data, x_sequence[i])
        y_data = np.append(y_data, y_sequence[i])

    return x_data, y_data


def creation_white_noise(N):
    x_data = np.random.rand(N)
    y_data = np.random.rand(N)

    x_data = ratio_0_to_100(x_data)
    y_data = ratio_0_to_100(y_data)

    return x_data, y_data


def pink_noise_x_y(N):
    """ This function creates a pink noise signal and then appends 1 value to the x_data and 1 value to the y_data"""
    pink_noise = cn.powerlaw_psd_gaussian(1, N * 2)
    pink_noise = ratio_0_to_1(pink_noise)
    x_data = []
    y_data = []

    for i in range(1, len(pink_noise), 2):
        y_data.append(pink_noise[i])
    for i in range(0, len(pink_noise), 2):
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
    pink_signal = cn.powerlaw_psd_gaussian(1, N)
    derivative = lib.derivative(pink_signal, 1)
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
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z

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
    cut_number = int(len(attractor) / desired_number)
    new_attractor = attractor[::cut_number]

    return new_attractor


def pink_noise_x_and_y(N):
    """ This function creates 2 different and seperate pink noise signals
    I suggest this approach but with a correction from the grip code"""
    x_data = pink_noise_signal_creation_using_cn(N)
    y_data = pink_noise_signal_creation_using_cn(N)
    x_data = ratio_0_to_100(x_data)
    y_data = ratio_0_to_100(y_data)

    return x_data, y_data


def pink_noise_signal_creation_using_cn(N):
    pink = False
    iterations = 0
    while pink == False:

        pink_noise = cn.powerlaw_psd_gaussian(1, N)

        slope, positive_freqs_log, positive_magnitude_log, intercept, name, r, p, positive_freqs, positive_magnitude = quality_assessment_of_temporal_structure_FFT_method(
            pink_noise, 'pink_noise_z')

        if round(slope, 2) == -0.5 and (p < 0.05) and (r <= -0.7):
            #
            # Figure of Frequincies vs Magnitude
            # plt.figure(figsize=(10,6))
            # plt.plot(positive_freqs, positive_magnitude)
            # plt.title(f'{name}\nFFT of Sine Wave')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Magnitude')
            # plt.grid()
            # plt.show()

            # plt.figure(figsize=(10, 6))
            # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
            # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept,
            #          label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
            # plt.title(f'{name}\nLog-Log Plot of FFT (Frequency vs Magnitude)')
            # plt.xlabel('Log(Frequency) (Hz)')
            # plt.ylabel('Log(Magnitude)')
            # plt.legend()
            # plt.grid()
            # plt.show()
            pink = True
        else:
            print('Not valid pink noise signal')
            iterations += 1
            print(iterations)

    return pink_noise


def quality_assessment_of_temporal_structure_FFT_method(signal, name):
    # Apply FFT
    fft_output = np.fft.fft(signal)  # FFT of the signal
    fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(len(signal), d=1 / 0.01)  # Frequency bins

    # Keep only the positive frequencies
    positive_freqs = frequencies[1:len(frequencies) // 2]  # Skip the zero frequency
    positive_magnitude = fft_magnitude[1:len(frequencies) // 2]  # Skip the zero frequency

    #  Figure of Frequincies vs Magnitude
    # plt.figure(figsize=(10,6))
    # plt.plot(positive_freqs, positive_magnitude)
    # plt.title(f'{name}\nFFT of Sine Wave')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.grid()
    # plt.show()

    positive_freqs_log = np.log10(positive_freqs[positive_freqs > 0])
    positive_magnitude_log = np.log10(positive_magnitude[positive_freqs > 0])

    r, p = pearsonr(positive_freqs_log, positive_magnitude_log)

    # Perform linear regression (best fit) to assess the slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(positive_freqs_log, positive_magnitude_log)
    print(f'r_value = {r_value}')
    print(f'p_value = {p_value}')

    # Plot the log-log results
    plt.figure(figsize=(10,6))
    plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
    plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept, label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
    plt.title(f'{name}\nLog-Log Plot of FFT (Frequency vs Magnitude)')
    plt.xlabel('Log(Frequency) (Hz)')
    plt.ylabel('Log(Magnitude)')
    plt.legend()
    plt.grid()
    plt.show()

    return slope, positive_freqs_log, positive_magnitude_log, intercept, name, r, p, positive_freqs, positive_magnitude


def convert_force_values(data_point):
    """
    This function takes data points of Kinvent force and returns force in kg
    """
    data_point = data_point / 10
    return data_point


def convert_quaternion(quaternion):
    """
    This function takes data points of Kinvent IMU quaternions and returns quaternion from -1 to 1
    """
    quaternion = (quaternion - 32768) / 16384
    return quaternion


def convert_acceleration(acc):
    """
    This function takes data points of Kinvent force and returns force in m/(s^2)
    """
    acc = (acc - 32768) * (16 / 32768)
    acc = acc * g
    return acc


def convert_ang_vel(vel):
    """
    This function takes data points of Kinvent IMU angular velocity and returns angular velocity in the range of +-2000°/s
    """
    vel = (vel - 32768) * (2000.0 / 32768)
    return vel


def convert_magnetic_field_density(mag):
    """
    This function takes data points of Kinvent IMU angular velocity and returns angular velocity in the range of +-2000°/s
    """
    mag = (mag - 32768) * (2500.0 / 32768)
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
    This way we only take the rows in which we have targets
    """

    filtered_data = df[df['target_pos_y'] != ' None']

    timestamp = filtered_data['timestamp'].to_numpy()
    target_pos_x = filtered_data['target_pos_x'].to_numpy()
    target_pos_y = filtered_data['target_pos_y'].to_numpy()
    player_pos_x = filtered_data['player_pos_x'].to_numpy()
    player_pos_y = filtered_data['player_pos_y'].to_numpy()
    left_plate = filtered_data['left plate'].to_numpy()
    right_plate = filtered_data['right plate'].to_numpy()
    pitch = filtered_data['pitch'].to_numpy()
    yaw = filtered_data['yaw'].to_numpy()
    roll = filtered_data['roll'].to_numpy()
    min_angle = filtered_data['min_angle'].to_numpy()
    max_angle = filtered_data['max_angle '].to_numpy()

    timestamp = converting_str_into_float(timestamp)
    target_pos_x = converting_str_into_float(target_pos_x)
    target_pos_y = converting_str_into_float(target_pos_y)
    player_pos_x = converting_str_into_float(player_pos_x)
    player_pos_y = converting_str_into_float(player_pos_y)
    left_plate = converting_str_into_float(left_plate)
    right_plate = converting_str_into_float(right_plate)
    pitch = converting_str_into_float(pitch)
    yaw = converting_str_into_float(yaw)
    roll = converting_str_into_float(roll)
    min_angle = converting_str_into_float(min_angle)
    max_angle = converting_str_into_float(max_angle)

    dist = {'timestamp': timestamp,
            'target_pos_x': target_pos_x,
            'target_pos_y': target_pos_y,
            'player_pos_x': player_pos_x,
            'player_pos_y': player_pos_y,
            'left_plate': left_plate,
            'right_plate': right_plate,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'min_angle': min_angle,
            'max_angle': max_angle
            }
    new_df = pd.DataFrame(dist)

    new_df_2 = Unix_to_start_from_zero_time(new_df)
    return new_df_2


def converting_str_into_float(time_series):
    time_series = time_series.astype(float)
    return time_series

def convert_target_position_X_if_old_data(signal):
    for i in range(len(signal)):
        if signal[i] < 128:
            signal[i] = 128
        if signal[i] > 1920 - 128:
            signal[i] = 1920 - 128
    return signal

def convert_target_position_Y_if_old_data(signal):
    for i in range(len(signal)):
        if signal[i] < 128:
            signal[i] = 128
        if signal[i] > 1080 - 128:
            signal[i] = 1080 - 128
    return signal


def return_the_df_of_each_target_separated_by_set(df, old_data):
    """ This function returns the dataframe of each target separated and for each set. It returns a list
    with 5 sublists which include 30 dataframes each in respect to the 30 targets each set has"""

    list_with_list_of_indices, indices_before_change = find_the_last_moment_before_target_change_position(df['target_pos_x'])

    list_with_all_df_separated_by_set = []
    for list in list_with_list_of_indices:
        list_of_each_set = []
        for i in range(len(list)-1):
            start = list[i] + 1
            stop = list[i + 1] + 1

            timestamp = df['timestamp'][start:stop].to_numpy()
            target_pos_x = df['target_pos_x'][start:stop].to_numpy()
            target_pos_y = df['target_pos_y'][start:stop].to_numpy()
            player_pos_x = df['player_pos_x'][start:stop].to_numpy()
            player_pos_y = df['player_pos_y'][start:stop].to_numpy()
            left_plate = df['left_plate'][start:stop].to_numpy()
            right_plate = df['right_plate'][start:stop].to_numpy()
            pitch = df['pitch'][start:stop].to_numpy()
            yaw = df['yaw'][start:stop].to_numpy()
            roll = df['roll'][start:stop].to_numpy()
            min_angle = df['min_angle'][start:stop].to_numpy()
            max_angle = df['max_angle'][start:stop].to_numpy()

            if old_data == True:
                target_pos_x = convert_target_position_X_if_old_data(target_pos_x)
                target_pos_y = convert_target_position_Y_if_old_data(target_pos_y)


            dist = {'timestamp': timestamp,
                    'target_pos_x': target_pos_x,
                    'target_pos_y': target_pos_y,
                    'player_pos_x': player_pos_x,
                    'player_pos_y': player_pos_y,
                    'left_plate': left_plate,
                    'right_plate': right_plate,
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    'min_angle': min_angle,
                    'max_angle': max_angle
                    }
            df_temporary = pd.DataFrame(dist)
            list_of_each_set.append(df_temporary)
        list_with_all_df_separated_by_set.append(list_of_each_set)

    return list_with_all_df_separated_by_set


def simple_graph(list_of_all_data, *column_to_plot):

    list_of_set_positions = []
    list_of_set_positions.append(0)
    concatenated_df_list = []
    current_index = 0
    for dataframe_list in list_of_all_data:
        concatenated_df = pd.concat(dataframe_list, ignore_index=True)
        concatenated_df_list.append(concatenated_df)
        current_index = current_index + len(concatenated_df)
        list_of_set_positions.append(current_index)
    all_data = pd.concat(concatenated_df_list, ignore_index=True)

    for column in column_to_plot:
        plt.plot(all_data[column], label=column)

    for i in range(len(list_of_set_positions)-1):
        plt.axvline(x=list_of_set_positions[i], linestyle='--', c='k')
    plt.axvline(x=list_of_set_positions[-1], linestyle='--', c='k', label='set')
    plt.legend()
    plt.show()


def find_the_last_moment_before_target_change_position(target_pos_x):
    """ This Function returns the indices of the targets right before the targets change position.
        First it considered the number_of_data_point before the targets changes. This is because, between
        sets, the last target changes after the break. Thus, we consider the number_of_data_point and if
        this number is higher than the median of the list_number_of_data_point then this index is changes as
        the index of the previous target + the median of the list_number_of_data_point.
        Additionally, the index of the last target is calculated as the index of the previous target + the
        median of the list_number_of_data_point. In this way we keep the right indices of the last moment
        of target appearance.
        """
    indices_before_change = []
    number_of_data_point = 0
    list_number_of_data_point = []
    list_with_list_of_indices = []
    indices_before_change.append(-1)  # I do this so that in the return_the_values_before_target_change function I can separate each target

    # We create a list "list_number_of_data_point" with the total number of the indices which have the same target
    # (we will use this for the median number afterward
    for i in range(len(target_pos_x) - 1):
        if target_pos_x[i] != target_pos_x[i + 1]:
            list_number_of_data_point.append(number_of_data_point)
            number_of_data_point = 0
        else:
            number_of_data_point = number_of_data_point + 1
    # Here we are good!!!!!!!!!!!!!!!!!!!!!!!

    # We create a list with all the indices before the targets change
    for i in range(len(target_pos_x) - 1):
        if target_pos_x[i] != target_pos_x[i + 1]:
            indices_before_change.append(i)
    indices_before_change.append(int(indices_before_change[-1] + np.median(list_number_of_data_point)))


    # We create a list with the indices during which the set is changed
    # CAREFUL at these indices the target changes, so we don't have the index of the end of the last target before the set changes
    indices_to_insert = []
    j=0 # we use this factor do the insert function we have later
    for i in range(len(indices_before_change) - 1):
        if indices_before_change[i + 1] > indices_before_change[i] + np.median(list_number_of_data_point) + 10:
            indices_to_insert.append(i+1+j)
            j=j+1

    # Here we insert the index of the end of the last target before the set changes
    for i in indices_to_insert:
        indices_before_change.insert(i, int(indices_before_change[i - 1] + np.median(list_number_of_data_point)))  # This might give us an error in the future

    # Here we create 5 lists which are consisted of the 5 sets and append the lists to another list
    start = 0
    for i in range(len(indices_before_change) - 1):
        if indices_before_change[i + 1] > indices_before_change[i] + np.median(list_number_of_data_point) + 10:
            stop = i+1
            list_with_list_of_indices.append(indices_before_change[start:stop])
            start = i + 1
    list_with_list_of_indices.append(indices_before_change[start:])

    return list_with_list_of_indices, indices_before_change


def spatial_error_calculation(target_pos_x, target_pos_y, player_pos_x, player_pos_y):
    spatial_error = np.sqrt((player_pos_x - target_pos_x) ** 2 + (player_pos_y - target_pos_y) ** 2)

    return spatial_error


def spatial_error_average_for_each_target(df, time_window=500, time_between_each_sample=25):
    """
    This function calculates the average spatial error for overlapping windows with length equal to time_window. Then it returns the minimum of those averages.
    We consider that this minimum is the best performance of the participant for this target. Also, we consider that time_between_each_sample is equal to 25, however this is
    just a standard number. In another function we calculate this number.
    Parameter:
        df                          :   the df with the stored data of each target
        factor                      :   this is the time in milliseconds of the window length for the calculation of the average spatial error
        time_between_each_sample    :   this is the standard time between each measurement.
        """
    # Here we check if the sampling frequency is close to 40. Basically we check if between each measurement the time is 25 milliseconds
    list_difference_between_time_points = []
    for i in range(len(df['timestamp'])-1):
        list_difference_between_time_points.append(df['timestamp'][i+1]-df['timestamp'][i])
    array_difference_between_time_points = np.array(list_difference_between_time_points)

    # The 4 lines bellow are of use if there is a problem with the time intervals
    # for i in range(len(df['timestamp'])-1):
    #     print(f"{df['timestamp'][i]}        {list_difference_between_time_points[i]}")

    if np.any(array_difference_between_time_points > time_between_each_sample + 4) or np.any(array_difference_between_time_points < time_between_each_sample - 4):
        pass
        # print(np.max(array_difference_between_time_points))
        # print(np.min(array_difference_between_time_points))
        # plt.plot(list_difference_between_time_points)
        # plt.show()
        # raise ValueError(f'CHECK THE TIME INTERVALS. The time between each consecutive observation is not near {time_between_each_sample}, which means that we dont calculate the windows with the factor {time_window} correct')
    # End of the check

    # Here we calculate the sampling frequency
    sampling_frequency = 1000/time_between_each_sample

    # Here we calculate the window length for the calculation of the average spatial error
    window_length = int((time_window/1000)*sampling_frequency)

    # Here we calculate the average of spatial error of each window and then the min of those averages.
    # We consider that this min is the best performance for this window. We probably need to play with the factor.
    list_of_average_spatial_error = []
    list_of_time_stamp = []

    for i in range(len(df['target_pos_x']) - window_length):
        target_pos_x = df['target_pos_x'][i: i + window_length].to_numpy()
        target_pos_y = df['target_pos_y'][i: i + window_length].to_numpy()
        player_pos_x = df['player_pos_x'][i: i + window_length].to_numpy()
        player_pos_y = df['player_pos_y'][i: i + window_length].to_numpy()
        df_time = df['timestamp'][i: i + window_length].to_numpy()


        # calculate and append the average spatial error of each window
        spatial_error_of_window = spatial_error_calculation(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
        average_spatial_error_of_window = np.mean(spatial_error_of_window)
        list_of_average_spatial_error.append(average_spatial_error_of_window)

        # calculate and append the middle time point of each window
        time_stamp = ((df_time[-1] - df_time[0])/2) + df_time[0]
        list_of_time_stamp.append(time_stamp)

    # Here I calculate the min of the spatial errors and find its index to find when did this happen
    min_of_average_spatial_error = np.min(list_of_average_spatial_error)
    min_index = list_of_average_spatial_error.index(min(list_of_average_spatial_error))
    time_stamp_of_min_spatial_error = list_of_time_stamp[min_index]

    return min_of_average_spatial_error, time_stamp_of_min_spatial_error


def spatial_error_best_window(list_with_all_df_separated_by_set, plot=False, time_window=400, time_between_each_sample=25):
    # pd.set_option('display.float_format', '{:.0f}'.format)
    list_spatial_error_all_separated_by_set = []
    list_time_stamp_of_min_spatial_error_separated_by_set = []
    list_of_big_lists = []
    for dataframe_list in list_with_all_df_separated_by_set:
        list_spatial_error_each_set = []
        list_time_stamp_of_min_spatial_error = []
        for df in dataframe_list:
            min_spatial_error, time_stamp_of_min_spatial_error = spatial_error_average_for_each_target(df, time_window, time_between_each_sample)
            list_spatial_error_each_set.append(min_spatial_error)
            list_time_stamp_of_min_spatial_error.append(time_stamp_of_min_spatial_error)

        list_spatial_error_all_separated_by_set.append(list_spatial_error_each_set)
        list_time_stamp_of_min_spatial_error_separated_by_set.append(list_time_stamp_of_min_spatial_error)

    if plot:
        list_of_set_positions = []
        list_of_set_positions.append(0)
        list_of_all_spatial_errors =[]
        current_index = 0


        for list_of_each_set in list_spatial_error_all_separated_by_set:
            for i in list_of_each_set:
                list_of_all_spatial_errors.append(i)
            current_index = current_index + len(list_of_each_set)
            list_of_set_positions.append(current_index)

        # list_of_all_spatial_errors = np.array(list_of_all_spatial_errors)
        # list_of_all_spatial_errors = list_of_all_spatial_errors - np.mean(list_of_all_spatial_errors)
        # list_of_all_spatial_errors = np.cumsum(list_of_all_spatial_errors)
        time = np.linspace(0,len(list_of_all_spatial_errors),len(list_of_all_spatial_errors))
        # plt.plot(list_of_all_spatial_errors, c='red', label='Min Spatial Error')
        plt.scatter(time, list_of_all_spatial_errors, c='red', label='Min Spatial Error')

        for i in range(len(list_of_set_positions) - 1):
            plt.axvline(x=list_of_set_positions[i], linestyle='--', c='k')
        plt.axvline(x=list_of_set_positions[-1], linestyle='--', c='k', label='set')
        plt.ylim(0,1200)
        plt.legend()
        plt.show()


    return list_spatial_error_all_separated_by_set, list_time_stamp_of_min_spatial_error_separated_by_set


def create_excel_file(x_data, y_data, directory, name):
    dist = {'X coordinates': x_data,
            'Y coordinates': y_data}

    excel = pd.DataFrame(dist)
    excel.to_excel(fr'{directory}\{name}.xlsx')


def convert_excel_to_screen_size_targets(excel, old_data, x_screen_size=1920, y_screen_size=1080):
    """ This function takes the Excel file with the targets from 0-100 and returns
    as they appear on the screen """

    if old_data == False:
        x_screen_size = x_screen_size - 2 * 128
        y_screen_size = y_screen_size - 2 * 128


    target_singal_x = excel['X coordinates'].to_numpy()
    target_singal_y = excel['Y coordinates'].to_numpy()

    target_singal_x = target_singal_x * x_screen_size / 100
    target_singal_y = target_singal_y * y_screen_size / 100

    if old_data == False:
        target_singal_x = target_singal_x + 128
        target_singal_y = target_singal_y + 128


    if old_data == True:
        target_singal_x = convert_target_position_X_if_old_data(target_singal_x)
        target_singal_y = convert_target_position_Y_if_old_data(target_singal_y)

    target_singal_x = np.round(target_singal_x, decimals=0)
    target_singal_y = np.round(target_singal_y, decimals=0)


    return target_singal_x, target_singal_y


def graph_creation_target_vs_player(list_with_all_df_separated_by_set, x_screen_size=1920, y_screen_size=1080):
    """ This function creates a graph with a slider to visualize better the position of target vs
    player position"""

    concatenated_df_list = []
    for dataframe_list in list_with_all_df_separated_by_set:
        concatenated_df = pd.concat(dataframe_list, ignore_index=True)
        concatenated_df_list.append(concatenated_df)
    all_data = pd.concat(concatenated_df_list, ignore_index=True)

    target_pos_x = all_data['target_pos_x'].to_numpy()
    target_pos_y = all_data['target_pos_y'].to_numpy()
    player_pos_x = all_data['player_pos_x'].to_numpy()
    player_pos_y = all_data['player_pos_y'].to_numpy()


    initial_points = 1

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Plot initial points
    sc_player = ax.scatter(player_pos_x[:initial_points], player_pos_y[:initial_points], label='Player', color='blue')
    sc_target = ax.scatter(target_pos_x[:initial_points], target_pos_y[:initial_points], label='Signal', color='red')

    ax.set_xlim(-50, x_screen_size + 50)
    ax.set_ylim(-50, y_screen_size + 50)
    ax.legend()

    # Create a slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Data Points', 1, len(player_pos_x), valinit=initial_points, valstep=1)

    # Update function
    def update(val):
        num_points = int(slider.val)  # Get current slider value

        # Update scatter plot data
        sc_player.set_offsets(np.column_stack((player_pos_x[:num_points], player_pos_y[:num_points])))
        sc_target.set_offsets(np.column_stack((target_pos_x[:num_points], target_pos_y[:num_points])))

        fig.canvas.draw_idle()  # Refresh the figure

    slider.on_changed(update)  # Connect slider to update function

    plt.show()


def graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set, x_screen_size=1920, y_screen_size=1080, update_target=5):
    """ This function creates a graph with a slider to visualize better the position of target vs
    player position"""

    concatenated_df_list = []
    current_index = 0
    list_of_set_positions = []
    list_of_set_positions.append(current_index)
    list_of_length_of_each_target = []
    for dataframe_list in list_with_all_df_separated_by_set:
        for df in dataframe_list:
            list_of_length_of_each_target.append(len(df))

        concatenated_df = pd.concat(dataframe_list, ignore_index=True)
        concatenated_df_list.append(concatenated_df)
        current_index = current_index + len(concatenated_df)
        list_of_set_positions.append(current_index)

    all_data = pd.concat(concatenated_df_list, ignore_index=True)


    target_pos_x = all_data['target_pos_x'].to_numpy()
    target_pos_y = all_data['target_pos_y'].to_numpy()
    player_pos_x = all_data['player_pos_x'].to_numpy()
    player_pos_y = all_data['player_pos_y'].to_numpy()
    IMU_yaw = all_data['yaw'].to_numpy()
    left_plate = all_data['left_plate'].to_numpy()
    right_plate = all_data['right_plate'].to_numpy()

    # Initial points
    initial_points = 1

    # Create a figure and two subplots (vertical layout)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # Two subplots vertically
    fig = plt.gcf()
    plt.subplots_adjust(top=0.99)
    plt.subplots_adjust(bottom=0.15)  # Increase the space at the bottom (higher value gives more space)

    # Plot initial points in the first subplot (ax1)
    sc_player1 = ax1.scatter(player_pos_x[:initial_points], player_pos_y[:initial_points], label='Player', color='blue')
    sc_target1 = ax1.scatter(target_pos_x[:initial_points], target_pos_y[:initial_points], label='Signal', color='red')

    ax1.set_xlim(-50, x_screen_size + 50)
    ax1.set_ylim(-50, y_screen_size + 50)
    ax1.legend()

    ax2_right = ax2.twinx()  # This creates a new y-axis on the right side of ax2

    # Plot initial points in the second subplot (ax2)
    line2, = ax2.plot(IMU_yaw[:initial_points], color='green')
    line3, = ax2_right.plot(left_plate[:initial_points], color='red', lw=0.3)
    line4, = ax2_right.plot(right_plate[:initial_points], color='blue', lw=0.3)

    ax2.set_xlim(0, len(IMU_yaw))  # Full x-axis range for IMU_yaw
    ax2.set_ylim(np.min(IMU_yaw), np.max(IMU_yaw))  # Dynamic y-axis range for IMU_yaw (left side)

    ax2_right.set_ylim(np.min([left_plate, right_plate]), np.max(
        [left_plate, right_plate]))  # Dynamic y-axis range for left_plate and right_plate (right side)
    # ax2_legend = ax2.legend([line2], ['IMU Yaw'], loc='upper left', fontsize=10, frameon=True)

    # Custom legend for ax2_right (Left & Right Plate)
    ax2_right_legend = ax2_right.legend([line2, line3, line4], ['IMU Yaw', 'Left Plate', 'Right Plate'], loc='upper right', fontsize=10,
                                        frameon=True)


    # Create a slider
    ax_slider = plt.axes([0.05, 0.001, 0.9, 0.03])  # Lower the slider's position (reduce 'bottom')
    slider = Slider(ax_slider, 'Time', 1, len(player_pos_x), valinit=initial_points, valstep=1)

    step = 0.15
    # Define button positions
    ax_button_forward_one = plt.axes([0.1 + 3*step, 0.05, 0.08, 0.05])  # Right-bottom corner
    ax_button_backward_one = plt.axes([0.1 + 2*step, 0.05, 0.08, 0.05])  # Left of the forward button
    ax_button_forward_ten = plt.axes([0.1 + 4*step, 0.05, 0.08, 0.05])  # Right-bottom corner
    ax_button_backward_ten = plt.axes([0.1 + step, 0.05, 0.08, 0.05])  # Left of the forward button
    ax_button_forward_one_target = plt.axes([0.1 + 5*step, 0.05, 0.08, 0.05])  # Right-bottom corner
    ax_button_backward_one_target = plt.axes([0.1, 0.05, 0.08, 0.05])  # Left of the forward button

    # Create buttons
    button_forward_one = Button(ax_button_forward_one, '+1')
    button_backward_one = Button(ax_button_backward_one, '-1')
    button_forward_ten = Button(ax_button_forward_ten, '+10')
    button_backward_ten = Button(ax_button_backward_ten, '-10')
    button_forward_one_target = Button(ax_button_forward_one_target, '+1 target')
    button_backward_one_target = Button(ax_button_backward_one_target, '-1 target')

    for i in range(len(list_of_set_positions) - 1):
        ax2.axvline(x=list_of_set_positions[i], linestyle='--', c='k')  # Draw on ax2
    ax2.axvline(x=list_of_set_positions[-1], linestyle='--', c='k', label='set')
    erase_rate = []
    erase_rate = [sum(list_of_length_of_each_target[i:i+update_target]) for i in range(0, len(list_of_length_of_each_target), update_target)]
    median_length_of_each_targe = np.median(list_of_length_of_each_target)

    # Update function
    def update(val):
        num_points = int(slider.val)  # Get current slider value
        erase_rate_index = min(num_points // erase_rate[0], len(erase_rate) - 1)  # Choose the appropriate step
        current_erase_rate = erase_rate[erase_rate_index]

        start_idx = (num_points // current_erase_rate) * current_erase_rate  # Compute the start index dynamically
        # Update scatter plot data for ax1 (show points only within the moving 100-value window)
        sc_player1.set_offsets(
            np.column_stack((player_pos_x[start_idx:num_points], player_pos_y[start_idx:num_points])))
        sc_target1.set_offsets(
            np.column_stack((target_pos_x[start_idx:num_points], target_pos_y[start_idx:num_points])))

        # Update the line plot data for ax2 (progressively display IMU_yaw, left_plate, and right_plate)
        line2.set_data(np.arange(num_points), IMU_yaw[:num_points])
        line3.set_data(np.arange(num_points), left_plate[:num_points])
        line4.set_data(np.arange(num_points), right_plate[:num_points])

        fig.canvas.draw_idle()  # Refresh the figure

    def move_forward_one(event):
        new_val = min(slider.val + 1, slider.valmax)  # Move forward
        slider.set_val(new_val)

    def move_backward_one(event):
        new_val = max(slider.val - 1, slider.valmin)  # Move backward
        slider.set_val(new_val)

    def move_forward_ten(event):
        new_val = min(slider.val + 10, slider.valmax)  # Move forward
        slider.set_val(new_val)

    def move_backward_ten(event):
        new_val = max(slider.val - 10, slider.valmin)  # Move backward
        slider.set_val(new_val)

    def move_forward_one_target(event):
        new_val = min(slider.val + median_length_of_each_targe, slider.valmax)  # Move forward
        slider.set_val(new_val)

    def move_backward_one_target(event):
        new_val = max(slider.val - median_length_of_each_targe, slider.valmin)  # Move backward
        slider.set_val(new_val)

    slider.on_changed(update)
    # Connect buttons to functions
    button_forward_one.on_clicked(move_forward_one)
    button_backward_one.on_clicked(move_backward_one)
    button_forward_ten.on_clicked(move_forward_ten)
    button_backward_ten.on_clicked(move_backward_ten)
    button_forward_one_target.on_clicked(move_forward_one_target)
    button_backward_one_target.on_clicked(move_backward_one_target)
    # plt.tight_layout()


    plt.show()


def graph_creation_of_spatial_error(target_pos_x, target_pos_y, player_pos_x, player_pos_y):
    """ This function creates a graph with separated sets of the spatial error"""

    time = np.linspace(1, 150, 150)
    spatial_error = spatial_error_calculation(target_pos_x, target_pos_y, player_pos_x, player_pos_y)

    plt.scatter(time, spatial_error, label='Spatial Error')
    plt.plot(time, spatial_error, linewidth=0.5)

    v_lines = [1, 30, 60, 90, 120, 150]
    for x in v_lines:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=2, alpha=0.7)

    plt.axvline(x=v_lines[-1], label='Sets', color='red', linestyle='--', linewidth=2, alpha=0.7)

    plt.ylim(0, 1000)
    plt.legend()
    plt.show()


def list_of_five_lists_flatten_list(nested_list):
    """ This function takes a nested list with 5 sublists and returns one flaten list with the indexes where those lists change to the next list
    CAUTION! this works only for nested lists which have sublists with data points and not dataframes"""
    flattened_list = list(itertools.chain(*nested_list))
    list_indices = [0]
    index = 0
    for sublist in nested_list:
        index = index + len(sublist)
        list_indices.append(index)
    flattened_list = np.array(flattened_list)
    list_indices = np.array(list_indices)

    return flattened_list, list_indices


def Unix_to_start_from_zero_time(df):
    first_time_point = df['timestamp'][0]
    df['timestamp'] = df['timestamp'] - first_time_point

    return df


def calculation_of_time_between_each_consecutive_data_point(list_with_all_df_separated_by_set, plot=True):
    all_list = []
    for list_ in list_with_all_df_separated_by_set:
        for df in list_:
            list_difference_between_time_points = []
            for i in range(len(df['timestamp']) - 1):
                list_difference_between_time_points.append(df['timestamp'][i + 1] - df['timestamp'][i])
            all_list.append(list_difference_between_time_points)
    flattened_list = list(itertools.chain(*all_list))
    max = np.max(flattened_list)
    min = np.min(flattened_list)
    mean = np.mean(flattened_list)
    std = np.std(flattened_list)
    CV = (std/mean)*100

    if plot == True:
        plt.plot(flattened_list)
        plt.show()
    return max, min, mean, std, CV


def creation_pf_timestamps_without_space_between_sets(list_time_stamp_of_min_spatial_error_separated_by_set):
    differences_in_time_between_targets = []
    for time_list in list_time_stamp_of_min_spatial_error_separated_by_set:
        time_list = np.array(time_list)
        differences = np.diff(time_list)
        differences_in_time_between_targets.append(differences)
    differences_in_time_between_targets, _ = list_of_five_lists_flatten_list(differences_in_time_between_targets)
    differences_in_time_between_targets = np.array(differences_in_time_between_targets)
    average_difference = np.mean(differences_in_time_between_targets)
    time_stamps_without_between_set_space = []
    last_time_stamp = 0
    for time_list in list_time_stamp_of_min_spatial_error_separated_by_set:
        initial_time = time_list[0]
        for time_stamp in time_list:
            time_stamp = time_stamp - initial_time + last_time_stamp
            time_stamps_without_between_set_space.append(time_stamp)
        last_time_stamp = time_stamp + average_difference
    time_stamps_without_between_set_space = np.array(time_stamps_without_between_set_space)

    return time_stamps_without_between_set_space


def simple_linear_regression(time_stamps_without_between_set_space, spatial_error, plot=False):
    slope, intercept, r_value, p_value, std_err = linregress(time_stamps_without_between_set_space, spatial_error)
    predicted_values = intercept + slope * np.array(time_stamps_without_between_set_space)

    rmse = np.sqrt(np.mean((spatial_error - predicted_values) ** 2))
    # Plot the original data

    if plot == True:
        plt.scatter(time_stamps_without_between_set_space, spatial_error, label="Original Data", color='blue', alpha=0.6)
        plt.plot(time_stamps_without_between_set_space, spatial_error, color='blue', alpha=0.6)

        plt.plot(time_stamps_without_between_set_space, predicted_values, label="Segmented Fit", linewidth=2, c='red')
        set_time_stamps = []
        for i in [0, 29, 59, 89, 119, 148]:
            set_time_stamps.append(time_stamps_without_between_set_space[i])
        for i in set_time_stamps:
            plt.axvline(x=i, linestyle='--', c='k')
        plt.xlabel("Time Stamps")
        plt.ylabel("Spatial Error")
        plt.legend()
        # plt.ylim(0, 800)
        plt.title(f'Linear Regression\nRMSE = {rmse}')
        plt.show()

    return slope, intercept, rmse


def segmented_linear_regression(time_stamps_without_between_set_space, spatial_error, number_of_breakpoints = 1, index_duration=15, plot=False):
    model = pwlf.PiecewiseLinFit(time_stamps_without_between_set_space, spatial_error)
    min_index = index_duration
    max_index = len(time_stamps_without_between_set_space) - index_duration

    x_min, x_max = time_stamps_without_between_set_space[min_index], time_stamps_without_between_set_space[max_index]

    breakpoints = model.fit(number_of_breakpoints + 1, bounds=[(x_min, x_max)])

    y_pred = model.predict(time_stamps_without_between_set_space)

    slopes = model.slopes
    intercepts = model.intercepts

    # Calculate R^2
    residuals = spatial_error - y_pred
    ss_residual = np.sum(residuals ** 2)
    ss_total = np.sum((spatial_error - np.mean(spatial_error)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calculate Adjusted R-squared
    n = len(spatial_error)
    p = len(breakpoints) + 1
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1)

    # Calculate RMSE
    rmse = np.sqrt(np.sum((y_pred - spatial_error) ** 2) / len(spatial_error))

    if plot == True:
        plt.scatter(time_stamps_without_between_set_space, spatial_error, label="Original Data", color='blue', alpha=0.6)
        plt.plot(time_stamps_without_between_set_space, y_pred, c='red', label="Segmented Fit", linewidth=2)

        # Plot vertical blue dashed lines for breakpoints

        plt.axvline(x=breakpoints[1], color='orange', linestyle='--', label="Breakpoint",lw=2)

        set_time_stamps = []
        for i in [29, 59, 89, 119, 148]:
            set_time_stamps.append(time_stamps_without_between_set_space[i])
        for i in set_time_stamps:
            plt.axvline(x=i, linestyle='--', c='k', lw=0.7)
        plt.axvline(x=0, linestyle='--', c='k', label='Set', lw=0.7)

        plt.xlabel("Time Stamps")
        plt.ylabel("Spatial Error")
        plt.title(f"Segmented Linear Regression\nRMSE = {rmse}")
        plt.ylim(0, 800)
        plt.legend()
        plt.show()

    return slopes, intercepts, rmse


def determine_the_number_of_breakpoints(time_stamps_without_between_set_space, spatial_error, max_number_of_breakpoints_to_check=15, index_duration=15):
    """
    Thus function determines the optimal number of breakpoints for the segmented regression model.
    The two main parameter of the model which will be affected by the number of breakpoints is the Goodness and Complexity of the model.
    There are two main methodologies to access the number of breakpoints, the Akaike Information Criterion (AIC) and the
    Bayesian Information Criterion (BIC). There main differences is that BIC applies stronger penalty to complexity, thus favoring more simple (with fewer breakpoints) models.

    This function returns the proposed number of breakpoints based on both AIC and BIC.
    """
    aic_values = []
    bic_values = []
    num_breakpoints = list(range(1, max_number_of_breakpoints_to_check))  # Test 1 to 2 breakpoints (you can adjust this range)

    for n in num_breakpoints:
        model = pwlf.PiecewiseLinFit(time_stamps_without_between_set_space, spatial_error)
        min_index = index_duration
        max_index = len(time_stamps_without_between_set_space) - index_duration

        x_min, x_max = time_stamps_without_between_set_space[min_index], time_stamps_without_between_set_space[max_index]

        # Corrected bounds format: list of tuples [(min, max)]
        breakpoint = model.fit(n + 1, bounds=[(x_min, x_max)])

        residuals = spatial_error - model.predict(time_stamps_without_between_set_space)
        rss = np.sum(residuals ** 2)  # Residual sum of squares
        n_params = 2 * n + 2  # Number of parameters: breakpoints + 2 (slope and intercept)

        # Compute AIC and BIC
        n_data = len(time_stamps_without_between_set_space)
        aic = 2 * n_params + n_data * np.log(rss / n_data)
        bic = n_params * np.log(n_data) + n_data * np.log(rss / n_data)

        aic_values.append(aic)
        bic_values.append(bic)

    optimal_aic_n = num_breakpoints[np.argmin(aic_values)]
    optimal_bic_n = num_breakpoints[np.argmin(bic_values)]

    print(f"Optimal number of breakpoints (AIC): {optimal_aic_n}")
    print(f"Optimal number of breakpoints (BIC): {optimal_bic_n}")

    return optimal_aic_n, optimal_bic_n


def asymptotes(time_stamps_without_between_set_space, spatial_error):
    A_ = spatial_error[0]
    C_ = np.min(spatial_error)
    print(C_)
    print(A_)
    # p0=[A_, 0.00000001, C_]
    def f(x, a, b, c):
        return a * (b ** x) + c

    def estimate_B_all(time_stamps_without_between_set_space, spatial_error, C):
        x_data = time_stamps_without_between_set_space
        y_data = spatial_error
        B_values = []
        for i in range(len(x_data) - 1):  # Loop through consecutive pairs
            x1, x2 = x_data[i], x_data[i + 1]
            y1, y2 = y_data[i], y_data[i + 1]

            # Ensure valid log computation (y - C must be positive)
            if y1 > C and y2 > C:
                B_i = (np.log(y1 - C) - np.log(y2 - C)) / (x2 - x1)
                B_values.append(B_i)
        print(np.mean(B_values))
        print(np.mean(np.abs(B_values)))
        t = np.arange(0,len(B_values),1)
        plt.plot(B_values)
        plt.scatter(t,B_values)

        plt.show()
        print(np.mean(B_values))
        return np.mean(B_values), np.std(B_values) if B_values else None  # Return average or None if no valid B

    B_init, B_sd = estimate_B_all(time_stamps_without_between_set_space, spatial_error, C_)
    print(f'B_init = {B_init} +- {B_sd}')
    print(np.abs(B_init))
    bounds = ([np.min(spatial_error) - 100, np.abs(B_init)/10, np.min(spatial_error) - 100], [np.max(spatial_error) + 100, np.abs(B_init)*10, np.max(spatial_error) + 100])
    params, _ = curve_fit(f, time_stamps_without_between_set_space, spatial_error, check_finite=True, p0=[np.mean(spatial_error), B_init, np.mean(spatial_error)], bounds=bounds)
    A, B, C = params
    print(f'y = {A}* {B}^x + {C}')
    y_line = f(time_stamps_without_between_set_space, A, B, C)

    plt.plot(time_stamps_without_between_set_space, y_line, '--', color='green', label='fit', lw=3)
    plt.axhline(y=C, c='k', label='Asymptote')
    plt.scatter(time_stamps_without_between_set_space, spatial_error)
    plt.legend()
    plt.show()


def custom_segmented_regression(time_stamps_without_between_set_space, spatial_error, minimum_targets, plot=False):
    rmse_list = []
    sse_list = []
    braekpoint_list = []
    part_1_list = []
    part_2_list = []
    time_part_1_list = []
    time_part_2_list = []
    predicted_values_part_1_list = []
    predicted_values_part_2_list = []
    slope_part_1_list = []
    intercept_part_1_list = []
    r_value_part_1_list = []
    p_value_part_1_list = []
    std_err_part_1_list = []
    rmse_part_1_list = []
    sse_part_1_list = []
    slope_part_2_list = []
    intercept_part_2_list = []
    r_value_part_2_list = []
    p_value_part_2_list = []
    std_err_part_2_list = []
    rmse_part_2_list = []
    sse_part_2_list = []


    for i in range(minimum_targets, len(spatial_error) - minimum_targets):
        part_1 = spatial_error[:i]
        part_2 = spatial_error[i:]
        time_part_1 = time_stamps_without_between_set_space[:i]
        time_part_2 = time_stamps_without_between_set_space[i:]

        slope_part_1 , intercept_part_1, r_value_part_1, p_value_part_1, std_err_part_1 = linregress(time_part_1, part_1)
        predicted_values_part_1 = intercept_part_1 + slope_part_1 * np.array(time_part_1)
        residuals_part_1 = part_1 - predicted_values_part_1
        sse_part_1 = np.sum(residuals_part_1 ** 2)
        rmse_part_1 = np.sqrt(np.sum((predicted_values_part_1 - part_1) ** 2) / len(part_1))




        slope_part_2, intercept_part_2, r_value_part_2, p_value_part_2, std_err_part_2 = linregress(time_part_2, part_2)
        predicted_values_part_2 = intercept_part_2 + slope_part_2 * np.array(time_part_2)
        residuals_part_2 = part_2 - predicted_values_part_2
        sse_part_2 = np.sum(residuals_part_2 ** 2)
        rmse_part_2 = np.sqrt(np.sum((predicted_values_part_2 - part_2) ** 2) / len(part_2))

        predictive_total_line = np.concatenate((predicted_values_part_1, predicted_values_part_2))
        total_residuals = spatial_error - predictive_total_line
        sse_total = np.sum(total_residuals ** 2)
        rmse_total = np.sqrt(np.sum((predictive_total_line - spatial_error) ** 2) / len(spatial_error))

        part_1_list.append(part_1)
        part_2_list.append(part_2)
        time_part_1_list.append(time_part_1)
        time_part_2_list.append(time_part_2)
        predicted_values_part_1_list.append(predicted_values_part_1)
        predicted_values_part_2_list.append(predicted_values_part_2)
        braekpoint_list.append(i)
        sse_list.append(sse_total)
        rmse_list.append(rmse_total)
        slope_part_1_list.append(slope_part_1)
        intercept_part_1_list.append(intercept_part_1)
        r_value_part_1_list.append(r_value_part_1)
        p_value_part_1_list.append(p_value_part_1)
        std_err_part_1_list.append(std_err_part_1)
        rmse_part_1_list.append(rmse_part_1)
        sse_part_1_list.append(sse_part_1)
        slope_part_2_list.append(slope_part_2)
        intercept_part_2_list.append(intercept_part_2)
        r_value_part_2_list.append(r_value_part_2)
        p_value_part_2_list.append(p_value_part_2)
        std_err_part_2_list.append(std_err_part_2)
        rmse_part_2_list.append(rmse_part_2)
        sse_part_2_list.append(sse_part_2)



    min_sse = min(sse_list)
    min_sse_index = sse_list.index(min_sse)


    # min_rmse = min(rmse_list)
    # min_rmse_index = rmse_list.index(min_rmse)

    rmse = rmse_list[min_sse_index]
    part_1 = part_1_list[min_sse_index]
    part_2 = part_2_list[min_sse_index]
    time_part_1 = time_part_1_list[min_sse_index]
    time_part_2 = time_part_2_list[min_sse_index]
    predicted_values_part_1 = predicted_values_part_1_list[min_sse_index]
    predicted_values_part_2 = predicted_values_part_2_list[min_sse_index]
    breakpoint = braekpoint_list[min_sse_index]
    slope_part_1 = slope_part_1_list[min_sse_index]
    intercept_part_1 = intercept_part_1_list[min_sse_index]
    r_value_part_1 = r_value_part_1_list[min_sse_index]
    p_value_part_1 = p_value_part_1_list[min_sse_index]
    std_err_part_1 = std_err_part_1_list[min_sse_index]
    rmse_part_1 = rmse_part_1_list[min_sse_index]
    slope_part_2 = slope_part_2_list[min_sse_index]
    intercept_part_2 = intercept_part_2_list[min_sse_index]
    r_value_part_2 = r_value_part_2_list[min_sse_index]
    p_value_part_2 = p_value_part_2_list[min_sse_index]
    std_err_part_2 = std_err_part_2_list[min_sse_index]
    rmse_part_2 = rmse_part_2_list[min_sse_index]


    if plot == True:
        plt.scatter(time_stamps_without_between_set_space, spatial_error, label="Original Data", color='blue', alpha=0.6)
        plt.plot(time_part_1, predicted_values_part_1, c='red', label="Segmented Fit", linewidth=2)
        plt.plot(time_part_2, predicted_values_part_2, c='red', linewidth=2)

        # Plot vertical blue dashed lines for breakpoints

        plt.axvline(x=time_stamps_without_between_set_space[breakpoint], color='orange', linestyle='--', label="Breakpoint",lw=2)

        set_time_stamps = []
        for i in [29, 59, 89, 119, 149]:
            set_time_stamps.append(time_stamps_without_between_set_space[i])
        for i in set_time_stamps:
            plt.axvline(x=i, linestyle='--', c='k', lw=0.7)
        plt.axvline(x=0, linestyle='--', c='k', label='Set', lw=0.7)

        plt.xlabel("Time Stamps")
        plt.ylabel("Spatial Error")
        plt.title(f"Custom Segmented Linear Regression\nRMSE = {rmse}")
        plt.ylim(0, 800)
        plt.legend()
        plt.show()


def asymptotes_2(time_stamps_without_between_set_space, spatial_error):
    def f(x, a, b, c, d):
        return a * (b ** (-d*x)) + c

    popt, _ = curve_fit(f, time_stamps_without_between_set_space, spatial_error, bounds=((0, 0.01, -np.inf, 0), (np.inf, np.inf, np.inf, 1)), maxfev=30000)
    a, b, c, d = popt
    x_line = time_stamps_without_between_set_space
    y_line = f(time_stamps_without_between_set_space, a, b, c, d)


    plt.plot(x_line, y_line, '--', color='green', label='fit')
    plt.axhline(y=c, c='k', label='Asymptote')

    plt.scatter(time_stamps_without_between_set_space, spatial_error)
    plt.legend()
    plt.show()


def big_list_to_5df_list(big_list):
    """ This Function takes a list which have all data separated to 5 elements, and every element consists of sublists with n dataframes,
    according to each target. It returns a list with those dfs as one for each of the 5 elements """
    df_5_list = []
    for list in big_list:
        combined_df = pd.concat(list, ignore_index=True)
        df_5_list.append(combined_df)
    return df_5_list
