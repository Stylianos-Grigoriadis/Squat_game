import Lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob
import seaborn as sns
import pwlf
import piecewise_regression
from scipy.stats import linregress
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
# N = 100
# time = np.linspace(0, 60, N)
# pink = lbs.pink_noise_signal_creation_using_cn(N)
# pink = lbs.ratio_0_to_100(pink)
#
# white = lbs.white_noise_signal_creation_using_FFT_method(N)
# # white = lbs.ratio_0_to_100(white)
#
# sine = lbs.sine_wave_signal_creation(N, 5)
# color_list = ["#FFC0CB", "#E7E6E6", "k"]
# name_list = ["Pink Noise", "White Noise", "Sine Wave"]
# signal_list = [pink, white, sine]
#
# for i in range(3):
#     print(i)
#     fig, ax = plt.subplots(figsize=(8, 4), facecolor='#E7E6E6')
#     ax.set_facecolor('#E7E6E6')
#     ax.plot(time, signal_list[i], color='k', lw=7)
#     ax.plot(time, signal_list[i], color=color_list[i], lw=4)
#     # ax.set_title(name_list[i])
#     # ax.set_xlabel("Time [s]")
#     # ax.set_ylabel("Amplitude")
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid(True)
#     plt.show()

# plt.figure(figsize=(8, 4))
# plt.plot(time, white, color='k', lw=7)
# plt.plot(time, white, color='white', lw=4)
# plt.title("White Noise (1/f)")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(8, 4))
# plt.plot(time, pink, color='k', lw=7)
# plt.plot(time, pink, color='#FFC0CB', lw=4)
# plt.title("Pink Noise (1/f)")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()


spatial_error = True
if spatial_error:

    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Results'
    os.chdir(directory)
    df = pd.read_excel(r'Results after exclusion of outliers.xlsx')
    # First, filter the DataFrame for each group
    df_pink = df[df['ID'] == 'Structured']
    df_repeated = df[df['ID'] == 'Non-variable']
    df_white = df[df['ID'] == 'Non-structured']

    # Then, calculate means for each set and assign to variables

    # Pink Noise
    Pink_Noise_average_Spatial_error_Set_1 = df_pink['Average Spatial error set 1'].mean()
    Pink_Noise_average_Spatial_error_Set_2 = df_pink['Average Spatial error set 2'].mean()
    Pink_Noise_average_Spatial_error_Set_3 = df_pink['Average Spatial error set 3'].mean()
    Pink_Noise_average_Spatial_error_Set_4 = df_pink['Average Spatial error set 4'].mean()
    Pink_Noise_average_Spatial_error_Set_5 = df_pink['Average Spatial error set 5'].mean()

    # Repeated
    Repeated_average_Spatial_error_Set_1 = df_repeated['Average Spatial error set 1'].mean()
    Repeated_average_Spatial_error_Set_2 = df_repeated['Average Spatial error set 2'].mean()
    Repeated_average_Spatial_error_Set_3 = df_repeated['Average Spatial error set 3'].mean()
    Repeated_average_Spatial_error_Set_4 = df_repeated['Average Spatial error set 4'].mean()
    Repeated_average_Spatial_error_Set_5 = df_repeated['Average Spatial error set 5'].mean()

    # White Noise
    White_Noise_average_Spatial_error_Set_1 = df_white['Average Spatial error set 1'].mean()
    White_Noise_average_Spatial_error_Set_2 = df_white['Average Spatial error set 2'].mean()
    White_Noise_average_Spatial_error_Set_3 = df_white['Average Spatial error set 3'].mean()
    White_Noise_average_Spatial_error_Set_4 = df_white['Average Spatial error set 4'].mean()
    White_Noise_average_Spatial_error_Set_5 = df_white['Average Spatial error set 5'].mean()

    list_Pink_Noise_average_Spatial_error = [Pink_Noise_average_Spatial_error_Set_1, Pink_Noise_average_Spatial_error_Set_2, Pink_Noise_average_Spatial_error_Set_3, Pink_Noise_average_Spatial_error_Set_4, Pink_Noise_average_Spatial_error_Set_5]
    list_Repeated_average_Spatial_error = [Repeated_average_Spatial_error_Set_1, Repeated_average_Spatial_error_Set_2, Repeated_average_Spatial_error_Set_3, Repeated_average_Spatial_error_Set_4, Repeated_average_Spatial_error_Set_5]
    list_White_Noise_average_Spatial_error = [White_Noise_average_Spatial_error_Set_1, White_Noise_average_Spatial_error_Set_2, White_Noise_average_Spatial_error_Set_3, White_Noise_average_Spatial_error_Set_4, White_Noise_average_Spatial_error_Set_5]

    time = np.array([1, 2, 3, 4, 5])

    slope_pink, intercept_pink, _, _, _ = linregress(time, list_Pink_Noise_average_Spatial_error)
    slope_repeated, intercept_repeated, _, _, _ = linregress(time, list_Repeated_average_Spatial_error)
    slope_white, intercept_white, _, _, _ = linregress(time, list_White_Noise_average_Spatial_error)

    fit_pink = slope_pink * time + intercept_pink
    fit_repeated = slope_repeated * time + intercept_repeated
    fit_white = slope_white * time + intercept_white


    # plt.scatter(time, list_Pink_Noise_average_Spatial_error, label='Pink_Noise')
    # plt.scatter(time, list_Repeated_average_Spatial_error, label='Repeated')
    # plt.scatter(time, list_White_Noise_average_Spatial_error, label='White_Noise')
    # plt.plot(time, fit_pink, color='#FFC0CB', linestyle='--')
    # plt.plot(time, fit_repeated, color='#4F4F4F', linestyle='--')
    # plt.plot(time, fit_white, color='#D3D3D3', linestyle='--')
    # plt.legend()
    # plt.show()

    
    
    

    print(df.columns)



    df_long = df.melt(id_vars=['ID'],
                      value_vars=['Average Spatial error set 1', 'Average Spatial error set 2',
                                  'Average Spatial error set 3', 'Average Spatial error set 4',
                                  'Average Spatial error set 5'],
                      var_name='Set', value_name='Average Spatial Error')  # Ensure correct name



    # Rename set names for readability
    df_long['Set'] = df_long['Set'].str.replace('Average Spatial error set ', 'Set ')

    df_long['ID'] = pd.Categorical(df_long['ID'], categories=["Non-variable", "Structured", "Non-structured"], ordered=True)

    # Custom color palette
    custom_palette = {
        "Non-variable": "#4F4F4F",    # Dark gray (first)
        "Structured": "#FFC0CB",      # Soft pink (second)
        "Non-structured": "#D3D3D3"      # Light gray (third)
    }

    offsets = {"Non-variable": -0.266, "Structured": 0, "Non-structured": 0.266}

    # Create the boxplot with the correct order
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Set', y='Average Spatial Error', hue='ID', data=df_long, palette=custom_palette, showfliers=False)

    # Overlay regression lines
    plt.plot(time - 1 + offsets['Structured'], fit_pink, color="#FFC0CB", label='Structured Trend', linewidth=3, zorder=10)
    plt.plot(time - 1 + offsets['Structured'], fit_pink, color="k", linewidth=5, zorder=9)

    plt.plot(time - 1 + offsets['Non-variable'], fit_repeated, color="#4F4F4F", label='Non-variable Trend', linewidth=3, zorder=10)
    plt.plot(time - 1 + offsets['Non-variable'], fit_repeated, color="k", linewidth=5, zorder=9)

    plt.plot(time - 1 + offsets['Non-structured'], fit_white, color="#D3D3D3", label='Non-structured Trend', linewidth=3, zorder=10)
    plt.plot(time - 1 + offsets['Non-structured'], fit_white, color="k", linewidth=5, zorder=9)



    # Customize plot
    plt.title('Spatial Error Across Sets and Groups')
    plt.ylabel('Average Spatial Error')
    plt.xlabel('')

    plt.ylim(15, 275)
    # Get the current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Create custom order
    custom_labels = ["Non-variable", "Non-variable Trend", "Structured", "Structured Trend",  "Non-structured", "Non-structured Trend"]

    # Reorder handles based on your custom label order
    ordered_handles = [handles[labels.index(label)] for label in custom_labels]

    # Place the legend below the plot with two rows
    plt.legend(ordered_handles, custom_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.05),
               ncol=3,
               frameon=False)

    plt.tight_layout()
    plt.show()

    Sets = range(1, 6)
    plt.plot(Sets, list_Repeated_average_Spatial_error, label='Repeated', c='#4F4F4F', lw=3)
    plt.plot(Sets, list_Pink_Noise_average_Spatial_error, label='Pink', c='#FFC0CB', lw=3)
    plt.plot(Sets, list_White_Noise_average_Spatial_error, label='White', c='#D3D3D3', lw=3)
    plt.legend()
    plt.xticks(Sets)
    plt.xlabel('Set')
    plt.ylabel('Average Spatial Error')

    plt.show()






else:
    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Results'
    os.chdir(directory)
    df_learning = pd.read_excel(r'Results of Non-linear analysis.xlsx')

    print(df_learning.columns)


    df_long = df_learning.melt(id_vars=['ID'],
                               value_vars=['SaEn_travel_distance_set_1', 'SaEn_travel_distance_set_2',
                                           'SaEn_travel_distance_set_3', 'SaEn_travel_distance_set_4',
                                           'SaEn_travel_distance_set_5'],
                               var_name='Set', value_name='Sample Entropy')  # Ensure correct name

    # Rename set names for readability
    df_long['Set'] = df_long['Set'].str.replace('Sample Entropy ', 'Set ')
    df_long['ID'] = pd.Categorical(df_long['ID'], categories=["Non-variable", "Structured", "Non-structured"], ordered=True)

    # Create boxplot
    custom_palette = {
        "Non-variable": "#4F4F4F",  # Dark gray (first)
        "Structured": "#FFC0CB",  # Soft pink (second)
        "Non-structured": "#D3D3D3"  # Light gray (third)
    }
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Set', y='Sample Entropy', hue='ID', data=df_long, palette=custom_palette, showfliers=False)

    custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5']
    plt.xticks(ticks=np.arange(len(custom_labels)), labels=custom_labels)

    # Customize plot
    plt.title('Sample Entropy of TD Across Sets Between Groups')
    plt.xlabel('')
    plt.ylabel('Sample Entropy')
    plt.legend()
    plt.ylim(0.25, 1.7)


    # Show plot
    plt.show()

    df_long = df_learning.melt(id_vars=['ID'],
                               value_vars=['DFA_travel_distance_set_1', 'DFA_travel_distance_set_2',
                                           'DFA_travel_distance_set_3', 'DFA_travel_distance_set_4',
                                           'DFA_travel_distance_set_5'],
                               var_name='Set', value_name='DFA')  # Ensure correct name

    # Rename set names for readability
    df_long['Set'] = df_long['Set'].str.replace('DFA ', 'Set ')
    df_long['ID'] = pd.Categorical(df_long['ID'], categories=["Non-variable", "Structured", "Non-structured"], ordered=True)

    # Create boxplot
    custom_palette = {
        "Non-variable": "#4F4F4F",  # Dark gray (first)
        "Structured": "#FFC0CB",  # Soft pink (second)
        "Non-structured": "#D3D3D3"  # Light gray (third)
    }
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Set', y='DFA', hue='ID', data=df_long, palette=custom_palette,
                showfliers=False)  # Use correct column name
    custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5']
    plt.xticks(ticks=np.arange(len(custom_labels)), labels=custom_labels)

    # Customize plot
    plt.title('Detrended Fluctuation Analysis of TD Across Sets and Between Groups')
    plt.xlabel('')
    plt.ylabel('α exponent')
    plt.legend()
    plt.ylim(0.43, 0.93)

    # Show plot
    plt.show()










