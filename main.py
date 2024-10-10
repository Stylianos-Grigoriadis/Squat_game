import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import colorednoise as cn

import lib


def ratio_0_to_1(data_series):
    """ Takes a data series and converts it into values from 0 to 1"""
    data_series = np.array(data_series)
    # data_series = data_series - np.min(data_series)
    # data_series = data_series/np.max(data_series)
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
    return x_data, y_data

def pink_noise_and_derivative(N):
    pink_signal = cn.powerlaw_psd_gaussian(1,N)
    derivative = lib.derivative(pink_signal,1)
    derivative = list(derivative)
    derivative.append(derivative[-1])
    x_data = pink_signal
    y_data = derivative
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
    x_data = x
    y_data = y
    return x_data, y_data

x_data, y_data = lorenz_x_data_y_data(0.01, 75)

print(type(x_data))
plt.plot(x_data)
plt.show()
# x_data, y_data = pink_noise_x_y(75)

# x_data, y_data = pink_noise_travel_distance_and_orientation(75)

# x_data, y_data = pink_noise_and_derivative(75)







# x_data = np.random.rand(100)
# y_data = np.random.rand(100)




fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)


scatter = ax.scatter(x_data[:1], y_data[:1], label='Scatter Points')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)


ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Points', 1, len(x_data), valinit=1, valstep=1)

# Update function for the slider
def update(val):
    num_points = int(slider.val)
    scatter.set_offsets(np.c_[x_data[:num_points], y_data[:num_points]])
    fig.canvas.draw_idle()


slider.on_changed(update)

ax.legend()
plt.show()
