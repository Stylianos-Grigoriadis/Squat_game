import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
from fathon import fathonUtils as fu
import fathon
import lib

def ratio_0_to_100(data_series):
    """ Takes a data series and converts it into values from 0 to 100"""
    data_series = np.array(data_series)
    data_series = (data_series - np.min(data_series)) / (np.max(data_series) - np.min(data_series))*100

    return data_series


x_data = cn.powerlaw_psd_gaussian(1,75)
y_data = cn.powerlaw_psd_gaussian(1,75)
x_data = ratio_0_to_100(x_data)
y_data = ratio_0_to_100(y_data)


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)


scatter = ax.scatter(x_data[:1], y_data[:1], label='Scatter Points')
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 110)


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
