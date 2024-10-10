import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
from fathon import fathonUtils as fu
import fathon
import lib

def DFA(variable):
    a = fu.toAggregated(variable)
        #b = fu.toAggregated(b)

    pydfa = fathon.DFA(a)

    winSizes = fu.linRangeByStep(start=4, end=int(len(variable)/9))
    revSeg = True
    polOrd = 1

    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    H, H_intercept = pydfa.fitFlucVec()
    plt.plot(np.log(n), np.log(F), 'ro')
    plt.plot(np.log(n), H_intercept + H * np.log(n), 'k-', label='H = {:.2f}'.format(H))
    plt.xlabel('ln(n)', fontsize=14)
    plt.ylabel('ln(F(n))', fontsize=14)
    plt.title('DFA', fontsize=14)
    plt.legend(loc=0, fontsize=14)
    #plt.clf()
    plt.show()
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
    Orientation = cn.powerlaw_psd_gaussian(1, 1000)
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
    x_data = cn.powerlaw_psd_gaussian(1, N)
    y_data = cn.powerlaw_psd_gaussian(1, N)
    x_data = ratio_0_to_100(x_data)
    y_data = ratio_0_to_100(y_data)

    return x_data, y_data


x_data_1, y_data_1 = pink_noise_x_y(75)

x_data_2, y_data_2 = pink_noise_travel_distance_and_orientation(75)

x_data_3, y_data_3 = pink_noise_and_derivative(75)
# dict = {'X coordinates': x_data_3, 'Y coordinates': y_data_3}
# df = pd.DataFrame(dict)
# df.to_excel('Target Coordinates.xlsx')
# print(df)

x_data_4, y_data_4 = lorenz_x_data_y_data(0.01, 100000)
x_data_4 = erase_attractor_values(x_data_4, 100)
y_data_4 = erase_attractor_values(y_data_4, 100)

x_data_5, y_data_5 = aizawa_x_data_y_data(0.01, 100000)
x_data_5 = erase_attractor_values(x_data_5, 100)
y_data_5 = erase_attractor_values(y_data_5, 100)

x_data_6, y_data_6 = pink_noise_x_and_y(75)

x_data_lists = [x_data_1, x_data_2, x_data_3, x_data_4, x_data_5, x_data_6]
y_data_lists = [y_data_1, y_data_2, y_data_3, y_data_4, y_data_5, y_data_6]

# Create figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

# Maximize space for graphs by reducing space for sliders and setting tight layout
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95, hspace=0.4)

# Titles for each scatter plot
titles = ['Just a Pink Noise signal', 'Travel distance and Orientation', 'Pink Noise signal and Derivative', 'Lorenz Attractor', 'Aizawa Attractor', 'Random Signal']

# Initialize scatter plots with independent data series (starting with the first index)
scatters = []
for i in range(6):
    ax = axs[i // 2, i % 2]
    scatter = ax.scatter(x_data_lists[i][:1], y_data_lists[i][:1])  # Start with only the first point
    scatters.append(scatter)
    ax.set_xlim(-10, 110)  # Fix the x-axis from 0 to 100
    ax.set_ylim(-10, 110)  # Fix the y-axis from 0 to 100
    ax.set_title(titles[i])  # Set a unique title for each graph

# Create sliders for each graph in a two-column layout with reduced size
slider_axes = [
    plt.axes([0.1, 0.07, 0.35, 0.02]),  # Slider 1 (left)
    plt.axes([0.55, 0.07, 0.35, 0.02]), # Slider 2 (right)
    plt.axes([0.1, 0.05, 0.35, 0.02]),  # Slider 3 (left)
    plt.axes([0.55, 0.05, 0.35, 0.02]), # Slider 4 (right)
    plt.axes([0.1, 0.03, 0.35, 0.02]),  # Slider 5 (left)
    plt.axes([0.55, 0.03, 0.35, 0.02])  # Slider 6 (right)
]

sliders = []

# Initialize sliders (with a range from 1 to the length of each data series, default starting at 1)
for i in range(6):
    slider = Slider(slider_axes[i], f'Index {i+1}', 1, len(x_data_lists[i]), valinit=1, valstep=1)
    sliders.append(slider)

# Update function for sliders to control how many data points are displayed in the scatter plots
def update(val):
    for i in range(6):
        idx = int(sliders[i].val)  # Get the current slider value
        # Update scatter plot data
        scatters[i].set_offsets(np.c_[x_data_lists[i][:idx], y_data_lists[i][:idx]])  # Update both X and Y data
    fig.canvas.draw_idle()  # Redraw the plot

# Attach the update function to each slider
for slider in sliders:
    slider.on_changed(update)

# Show the plot
plt.show()