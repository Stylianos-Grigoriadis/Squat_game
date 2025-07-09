import Lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob
import pwlf
from scipy.stats import pearsonr
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
from matplotlib import font_manager
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16


def quality_assessment_of_temporal_structure_FFT_method(signal):
    # Apply FFT
    fft_output = np.fft.fft(signal)  # FFT of the signal
    fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(len(signal), d=1 / 0.01)  # Frequency bins

    # Keep only the positive frequencies
    positive_freqs = frequencies[1:len(frequencies) // 2]  # Skip the zero frequency
    positive_magnitude = fft_magnitude[1:len(frequencies) // 2]  # Skip the zero frequency

    positive_freqs_log = np.log10(positive_freqs[positive_freqs > 0])
    positive_magnitude_log = np.log10(positive_magnitude[positive_freqs > 0])

    r, p = pearsonr(positive_freqs_log, positive_magnitude_log)

    # Perform linear regression (best fit) to assess the slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(positive_freqs_log, positive_magnitude_log)

    # Plot the log-log results
    # plt.figure(figsize=(10,6))
    # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
    # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept, label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
    # plt.xlabel('Log(Frequency) (Hz)')
    # plt.ylabel('Log(Magnitude)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return slope, positive_freqs_log, positive_magnitude_log, intercept, r, p

def create_white_noise(number_of_data_points):
    white = False
    i = 0
    while white == False:
        data = np.random.rand(number_of_data_points)
        data = lbs.ratio_0_to_100(data)
        i +=1
        print(i)
        slope, positive_freqs_log, positive_magnitude_log, intercept, r, p = quality_assessment_of_temporal_structure_FFT_method(data)
        print(round(slope,1))
        if round(np.abs(slope), 2) == 0:
            white = True
            print('Hello')

    return data


# This is if I want to spice things up
os.chdir(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Figures')
img = mpimg.imread('target.png')
imagebox = OffsetImage(img, zoom=0.04)


number_of_data_points = 30
x_data_sine, y_data_sine = lbs.creation_rigid_signal(number_of_data_points)
x_data_white = create_white_noise(number_of_data_points)
y_data_white = create_white_noise(number_of_data_points)
x_data_pink, y_data_pink = lbs.pink_noise_x_and_y(number_of_data_points)

x_sine_slope, x_sine_positive_freqs_log, x_sine_positive_magnitude_log, x_sine_intercept, x_sine_r, x_sine_p = quality_assessment_of_temporal_structure_FFT_method(x_data_sine)
y_sine_slope, y_sine_positive_freqs_log, y_sine_positive_magnitude_log, y_sine_intercept, y_sine_r, y_sine_p = quality_assessment_of_temporal_structure_FFT_method(y_data_sine)
x_white_slope, x_white_positive_freqs_log, x_white_positive_magnitude_log, x_white_intercept, x_white_r, x_white_p = quality_assessment_of_temporal_structure_FFT_method(x_data_white)
y_white_slope, y_white_positive_freqs_log, y_white_positive_magnitude_log, y_white_intercept, y_white_r, y_white_p = quality_assessment_of_temporal_structure_FFT_method(y_data_white)
x_pink_slope, x_pink_positive_freqs_log, x_pink_positive_magnitude_log, x_pink_intercept, x_pink_r, x_pink_p = quality_assessment_of_temporal_structure_FFT_method(x_data_pink)
y_pink_slope, y_pink_positive_freqs_log, y_pink_positive_magnitude_log, y_pink_intercept, y_pink_r, y_pink_p = quality_assessment_of_temporal_structure_FFT_method(y_data_pink)

x_data_sine = lbs.Perc(x_data_sine, 1920, 0)
y_data_sine = lbs.Perc(y_data_sine, 1080, 0)
x_data_pink = lbs.Perc(x_data_pink, 1920, 0)
y_data_pink = lbs.Perc(y_data_pink, 1080, 0)
x_data_white = lbs.Perc(x_data_white, 1920, 0)
y_data_white = lbs.Perc(y_data_white, 1080, 0)




fig, axes = plt.subplots(2, 3, figsize=(15, 5))  # Adjust figsize as needed

font = font_manager.FontProperties(family='serif', size=12, weight='bold')


axes[0,0].scatter(x_sine_positive_freqs_log, x_sine_positive_magnitude_log, label='X axis', c='#2F2F2F')
axes[0,0].scatter(y_sine_positive_freqs_log, y_sine_positive_magnitude_log, label='Y axis', edgecolors='#2F2F2F', facecolors='white', marker='o')
axes[0,0].plot(x_sine_positive_freqs_log, x_sine_slope * x_sine_positive_freqs_log + x_sine_intercept, label=f'Slope = {x_sine_slope:.1f}', c='#2F2F2F', lw=3)
axes[0,0].plot(y_sine_positive_freqs_log, y_sine_slope * y_sine_positive_freqs_log + y_sine_intercept, label=f'Slope = {y_sine_slope:.1f}', c='#2F2F2F', linestyle='--')
axes[0,0].set_title("Non-variable group")
axes[0,0].set_xlabel("Log(Frequency)")
axes[0,0].set_ylabel("Log(Magnitude)\n")
axes[0,0].legend(frameon=False, prop=font)
axes[0,0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


axes[0,1].scatter(x_pink_positive_freqs_log, x_pink_positive_magnitude_log, label='X axis', c='#FF8FA3')
axes[0,1].scatter(y_pink_positive_freqs_log, y_pink_positive_magnitude_log, label='Y axis', edgecolors='#FF8FA3', facecolors='white', marker='o')
axes[0,1].plot(x_pink_positive_freqs_log, x_pink_slope * x_pink_positive_freqs_log + x_pink_intercept, label=f'Slope = {x_pink_slope:.1f}', c='#FF8FA3', lw=3)
axes[0,1].plot(y_pink_positive_freqs_log, y_pink_slope * y_pink_positive_freqs_log + y_pink_intercept, label=f'Slope = {y_pink_slope:.1f}', c='#FF8FA3', linestyle='--')
axes[0,1].set_title("Structured group")
axes[0,1].set_xlabel("Log(Frequency)")
axes[0,1].legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=False, prop=font)
axes[0,1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


axes[0,2].scatter(x_white_positive_freqs_log, x_white_positive_magnitude_log, label='X axis', c='#B0B0B0')
axes[0,2].scatter(y_white_positive_freqs_log, y_white_positive_magnitude_log, label='Y axis', edgecolors='#B0B0B0', facecolors='white', marker='o')
axes[0,2].plot(x_white_positive_freqs_log, x_white_slope * x_white_positive_freqs_log + x_white_intercept, label=f'Slope = {np.abs(x_white_slope):.1f}', c='#B0B0B0', lw=3)
axes[0,2].plot(y_white_positive_freqs_log, y_white_slope * y_white_positive_freqs_log + y_white_intercept, label=f'Slope = {np.abs(y_white_slope):.1f}', c='#B0B0B0', linestyle='--')
axes[0,2].set_title("Non-structured group")
axes[0,2].set_xlabel("Log(Frequency)")
axes[0,2].legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=False, prop=font)
axes[0,2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[0,2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))



for (x, y) in zip(x_data_sine, y_data_sine):
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    axes[1,0].add_artist(ab)
# axes[1,0].scatter(x_data_sine, y_data_sine, c='#2F2F2F', s=100, marker='x')
axes[1,0].set_xlabel("X coordinates")
axes[1,0].set_ylabel("Y coordinates")
axes[1,0].set_xlim(-80, 2000)
axes[1,0].set_ylim(-80, 1160)

for (x, y) in zip(x_data_pink, y_data_pink):
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    axes[1,1].add_artist(ab)
# axes[1,1].scatter(x_data_pink, y_data_pink, c='#FF8FA3', s=100, marker='x')
axes[1,1].set_xlabel("X coordinates")
axes[1,1].set_xlim(-80, 2000)
axes[1,1].set_ylim(-80, 1160)

for (x, y) in zip(x_data_white, y_data_white):
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    axes[1,2].add_artist(ab)
# axes[1,2].scatter(x_data_white, y_data_white, c='#D3D3D3', s=100, marker='x')
axes[1,2].set_xlabel("X coordinates")
axes[1,2].set_xlim(-80, 2000)
axes[1,2].set_ylim(-80, 1160)

# Optional: Adjust layout
plt.subplots_adjust(
    left=0.08,     # space from left edge of figure
    bottom=0.08,    # space from bottom edge
    right=0.98,    # space from right edge
    top=0.95,       # space from top edge
    wspace=0.17,    # width (horizontal) space between subplots
    hspace=0.3     # height (vertical) space between subplots
)
plt.show()

# Colors
# "Static Lighter"  : #6F6F6F
# "Static"          : #4F4F4F
# "Static Darker"   : #2F2F2F

# "Pink Lighter"  : #FFE4EC
# "Pink"          : #FFC0CB
# "Pink Darker"   : #FF8FA3

# "White Lighter"  : #E8E8E8
# "White"          : #D3D3D3
# "White Darker"   : #B0B0B0
