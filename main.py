import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
from fathon import fathonUtils as fu
import fathon
import lib
import lib_squats as lbs



number_of_data_points = 1000

example_signal_pink = cn.powerlaw_psd_gaussian(1, number_of_data_points)
example_signal_brown = cn.powerlaw_psd_gaussian(2, number_of_data_points)
print(f'example_signal_pink : {lbs.DFA(example_signal_pink)}')
print(f'example_signal_brown : {lbs.DFA(example_signal_brown)}')

x_data_1, y_data_1 = lbs.pink_noise_x_y(number_of_data_points)

x_data_2, y_data_2 = lbs.pink_noise_travel_distance_and_orientation(number_of_data_points)

x_data_3, y_data_3 = lbs.pink_noise_and_derivative(number_of_data_points)

x_data_4, y_data_4 = lbs.lorenz_x_data_y_data(0.01, 100000)
x_data_4 = lbs.erase_attractor_values(x_data_4, number_of_data_points)
y_data_4 = lbs.erase_attractor_values(y_data_4, number_of_data_points)

x_data_5, y_data_5 = lbs.aizawa_x_data_y_data(0.01, 100000)
x_data_5 = lbs.erase_attractor_values(x_data_5, number_of_data_points)
y_data_5 = lbs.erase_attractor_values(y_data_5, number_of_data_points)

x_data_6, y_data_6 = lbs.pink_noise_x_and_y(number_of_data_points)

dfa_x_data_1 = lbs.DFA(x_data_1)
dfa_y_data_1 = lbs.DFA(y_data_1)
dfa_x_data_2 = lbs.DFA(x_data_2)
dfa_y_data_2 = lbs.DFA(y_data_2)
dfa_x_data_3 = lbs.DFA(x_data_3)
dfa_y_data_3 = lbs.DFA(y_data_3)
dfa_x_data_4 = lbs.DFA(x_data_4)
dfa_y_data_4 = lbs.DFA(y_data_4)
dfa_x_data_5 = lbs.DFA(x_data_5)
dfa_y_data_5 = lbs.DFA(y_data_5)
dfa_x_data_6 = lbs.DFA(x_data_6)
dfa_y_data_6 = lbs.DFA(y_data_6)

print(f'DFA for subsequent values x axis: {dfa_x_data_1}')
print(f'DFA for subsequent values y axis: {dfa_y_data_1}')
print(f'DFA with pink travel distance and orientation x axis: {dfa_x_data_2}')
print(f'DFA with pink travel distance and orientation y axi: {dfa_y_data_2}')
print(f'DFA for pink noise signal and each derivative axis x: {dfa_x_data_3}')
print(f'DFA for pink noise signal and each derivative axis y: {dfa_y_data_3}')
print(f'DFA lorenz attractor axis x: {dfa_x_data_4}')
print(f'DFA lorenz attractor axis y: {dfa_y_data_4}')
print(f'DFA aizawa attractor axis x: {dfa_x_data_5}')
print(f'DFA aizawa attractor axis y: {dfa_y_data_5}')
print(f'DFA for two pink noise different signals axis x: {dfa_x_data_6}')
print(f'DFA for two pink noise different signals axis y: {dfa_y_data_6}')



x_data_lists = [x_data_1, x_data_2, x_data_3, x_data_4, x_data_5, x_data_6]
y_data_lists = [y_data_1, y_data_2, y_data_3, y_data_4, y_data_5, y_data_6]

# Create figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

# Maximize space for graphs by reducing space for sliders and setting tight layout
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95, hspace=0.4)

# Titles for each scatter plot
titles = ['Just a Pink Noise signal', 'Travel distance and Orientation', 'Pink Noise signal and Derivative', 'Lorenz Attractor', 'Aizawa Attractor', 'Pink noise x and y']

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

