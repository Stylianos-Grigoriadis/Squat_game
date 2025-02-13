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
from scipy import stats
from scipy.stats import pearsonr


def DFA(variable):
    a = fu.toAggregated(variable)
    # b = fu.toAggregated(b)

    pydfa = fathon.DFA(a)

    winSizes = fu.linRangeByStep(start=4, end=int(len(variable) / 4))
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
    for i in range(0,more):
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


        slope, positive_freqs_log, positive_magnitude_log, intercept, name, r, p, positive_freqs, positive_magnitude = quality_assessment_of_temporal_structure_FFT_method(pink_noise, 'pink_noise_z')

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
            iterations +=1
            print(iterations)

    return pink_noise


def quality_assessment_of_temporal_structure_FFT_method(signal, name):
    # Apply FFT
    fft_output = np.fft.fft(signal)  # FFT of the signal
    fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(len(signal), d=1/0.01)  # Frequency bins

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
    # plt.figure(figsize=(10,6))
    # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
    # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept, label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
    # plt.title(f'{name}\nLog-Log Plot of FFT (Frequency vs Magnitude)')
    # plt.xlabel('Log(Frequency) (Hz)')
    # plt.ylabel('Log(Magnitude)')
    # plt.legend()
    # plt.grid()
    # plt.show()

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
    target_pos_x = filtered_data['target_pos_x'].to_numpy()
    target_pos_y = filtered_data['target_pos_y'].to_numpy()
    player_pos_x = filtered_data['player_pos_x'].to_numpy()
    player_pos_y = filtered_data['player_pos_y'].to_numpy()
    left_plate = filtered_data['left plate'].to_numpy()
    right_plate = filtered_data['right plate'].to_numpy()
    pitch = filtered_data['pitch'].to_numpy()
    yaw = filtered_data['yaw'].to_numpy()
    roll = filtered_data['roll'].to_numpy()



    target_pos_x = converting_str_into_float(target_pos_x)
    target_pos_y = converting_str_into_float(target_pos_y)
    player_pos_x = converting_str_into_float(player_pos_x)
    player_pos_y = converting_str_into_float(player_pos_y)
    left_plate = converting_str_into_float(left_plate)
    right_plate = converting_str_into_float(right_plate)
    pitch = converting_str_into_float(pitch)
    yaw = converting_str_into_float(yaw)
    roll = converting_str_into_float(roll)



    return target_pos_x, target_pos_y, player_pos_x, player_pos_y, left_plate, right_plate, pitch, yaw, roll


def converting_str_into_float(time_series):
    time_series = time_series.astype(float)
    return time_series


def return_the_values_before_target_change(target_pos_x, target_pos_y, player_pos_x, player_pos_y):
    indexes_before_change = find_the_last_moment_before_target_change_position(target_pos_x)
    target_pos_x = target_pos_x[indexes_before_change]
    target_pos_y = target_pos_y[indexes_before_change]
    player_pos_x = player_pos_x[indexes_before_change]
    player_pos_y = player_pos_y[indexes_before_change]
    # print(len(target_pos_x))

    return target_pos_x, target_pos_y, player_pos_x, player_pos_y


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
    indexes_before_change = []
    number_of_data_point = 0
    list_number_of_data_point = []
    for i in range(len(target_pos_x) - 1):
        if target_pos_x[i] != target_pos_x[i + 1]:
            list_number_of_data_point.append(number_of_data_point)
            number_of_data_point = 0
        else:
            number_of_data_point = number_of_data_point + 1
    for i in range(len(target_pos_x) - 1):
        if target_pos_x[i] != target_pos_x[i + 1]:
            indexes_before_change.append(i)
    indexes_before_change.append(int(indexes_before_change[-1] + np.median(list_number_of_data_point)))

    for i in range(len(list_number_of_data_point)):
        if list_number_of_data_point[i] > np.median(list_number_of_data_point) + 10:
            indexes_before_change[i] = int(indexes_before_change[i-1] + np.median(list_number_of_data_point)) # This might give us an error in the future

    return indexes_before_change


def spatial_error_calculation(target_pos_x, target_pos_y, player_pos_x, player_pos_y):
    spatial_error = np.sqrt((player_pos_x - target_pos_x) ** 2 + (player_pos_y - target_pos_y) ** 2)

    return spatial_error


def create_excel_file(x_data, y_data, directory, name):
    dist = {'X coordinates': x_data,
            'Y coordinates': y_data}

    excel = pd.DataFrame(dist)
    excel.to_excel(fr'{directory}\{name}.xlsx')

def convert_excel_to_screen_size_targets(excel, x_screen_size=1920, y_screen_size=1080):
    """ This function takes the Excel file with the targets from 0-100 and returns
    as they appear on the screen """
    target_singal_x = excel['X coordinates'].to_numpy()
    target_singal_y = excel['Y coordinates'].to_numpy()

    target_singal_x = target_singal_x * x_screen_size / 100
    target_singal_y = target_singal_y * y_screen_size / 100

    return target_singal_x, target_singal_y


def graph_creation_target_vs_player(target_pos_x, target_pos_y, player_pos_x, player_pos_y, x_screen_size=1920, y_screen_size=1080):
    """ This function creates a graph with a slider to visualize better the position of target vs
    player position"""
    initial_points = 1

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    sc_player = ax.scatter(player_pos_x[:initial_points], player_pos_y[:initial_points], label='Player', color='blue')
    sc_target = ax.scatter(target_pos_x[:initial_points], target_pos_y[:initial_points], label='Signal', color='red')

    ax.set_xlim(-50, x_screen_size + 50)
    ax.set_ylim(-50, y_screen_size + 50)
    ax.legend()

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Data Points', 1, len(player_pos_x), valinit=initial_points, valstep=1)

    def update(val):
        num_points = int(slider.val)

        sc_player.set_offsets(np.column_stack((player_pos_x[:num_points], player_pos_y[:num_points])))
        sc_target.set_offsets(np.column_stack((target_pos_x[:num_points], target_pos_y[:num_points])))

        fig.canvas.draw_idle()

    slider.on_changed(update)

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

    plt.ylim(0,1000)
    plt.legend()
    plt.show()
