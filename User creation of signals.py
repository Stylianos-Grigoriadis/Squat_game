import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
import lib
import Lib_squats as lbs



number_of_data_points = 30
x_data_sine, y_data_sine = lbs.creation_rigid_signal(number_of_data_points)
x_data_random, y_data_random = lbs.creation_white_noise(number_of_data_points)
x_data_6, y_data_6 = lbs.pink_noise_x_and_y(number_of_data_points)


plt.scatter(x_data_sine, y_data_sine, label='sine', c='red', lw=5)
plt.scatter(x_data_random, y_data_random, label='random', c='k')
plt.scatter(x_data_6, y_data_6, label='pink', c='pink')
plt.legend()
plt.show()





