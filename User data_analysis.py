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
from matplotlib.widgets import CheckButtons

import lib

pd.set_option("display.max_rows", None)

participants_before_change = ['pink1', 'pink10', 'pink11', 'pink12', 'pink13', 'pink14','pink15', 'pink2', 'pink3', 'pink4', 'pink5', 'pink6', 'pink7', 'pink8', 'pink9', 'static1', 'static10', 'static11', 'static12', 'static13', 'static2', 'static3', 'static4', 'static5', 'static6', 'static7', 'static8', 'static9', 'white1', 'white10', 'white11', 'white12', 'white13', 'white14', 'white2', 'white3', 'white4', 'white5', 'white6', 'white7', 'white8', 'white9']


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data\Valid Data'
files = glob.glob(os.path.join(directory_path, "*"))

list_simple_rmse = []
list_simple_slope = []
list_simple_intercept = []

list_segmented_rmse = []
list_segmented_slope_before = []
list_segmented_intercept_before = []
list_segmented_slope_after = []
list_segmented_intercept_after = []

list_ID = []
list_exact_ID = []
list_average_spatial_error = []
list_sd_spatial_error = []
list_breakpoints = []
list_slope_before_change = []
list_slope_after_change = []

list_spatial_error_set_1_average = []
list_spatial_error_set_2_average = []
list_spatial_error_set_3_average = []
list_spatial_error_set_4_average = []
list_spatial_error_set_5_average = []

list_spatial_error_set_1_sd = []
list_spatial_error_set_2_sd = []
list_spatial_error_set_3_sd = []
list_spatial_error_set_4_sd = []
list_spatial_error_set_5_sd = []

list_power_90_set_1_x = []
list_power_90_set_2_x = []
list_power_90_set_3_x = []
list_power_90_set_4_x = []
list_power_90_set_5_x = []
list_power_90_set_1_y = []
list_power_90_set_2_y = []
list_power_90_set_3_y = []
list_power_90_set_4_y = []
list_power_90_set_5_y = []

list_power_95_set_1_x = []
list_power_95_set_2_x = []
list_power_95_set_3_x = []
list_power_95_set_4_x = []
list_power_95_set_5_x = []
list_power_95_set_1_y = []
list_power_95_set_2_y = []
list_power_95_set_3_y = []
list_power_95_set_4_y = []
list_power_95_set_5_y = []

list_power_99_set_1_x = []
list_power_99_set_2_x = []
list_power_99_set_3_x = []
list_power_99_set_4_x = []
list_power_99_set_5_x = []
list_power_99_set_1_y = []
list_power_99_set_2_y = []
list_power_99_set_3_y = []
list_power_99_set_4_y = []
list_power_99_set_5_y = []

list_SaEn_x_set_1 = []
list_SaEn_x_set_2 = []
list_SaEn_x_set_3 = []
list_SaEn_x_set_4 = []
list_SaEn_x_set_5 = []
list_SaEn_y_set_1 = []
list_SaEn_y_set_2 = []
list_SaEn_y_set_3 = []
list_SaEn_y_set_4 = []
list_SaEn_y_set_5 = []

list_DFA_x_set_1 = []
list_DFA_x_set_2 = []
list_DFA_x_set_3 = []
list_DFA_x_set_4 = []
list_DFA_x_set_5 = []
list_DFA_y_set_1 = []
list_DFA_y_set_2 = []
list_DFA_y_set_3 = []
list_DFA_y_set_4 = []
list_DFA_y_set_5 = []

list_SaEn_travel_distance_set_1 = []
list_SaEn_travel_distance_set_2 = []
list_SaEn_travel_distance_set_3 = []
list_SaEn_travel_distance_set_4 = []
list_SaEn_travel_distance_set_5 = []

list_DFA_travel_distance_set_1 = []
list_DFA_travel_distance_set_2 = []
list_DFA_travel_distance_set_3 = []
list_DFA_travel_distance_set_4 = []
list_DFA_travel_distance_set_5 = []





for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    list_exact_ID.append(ID)
    print(ID)
    if ID in participants_before_change:
        old_data = True
        print('Data taken before the fix')
    else:
        old_data = False
        print('Data taken after the fix')



    if 'static' not in ID:
        targets = pd.read_excel(rf'{ID}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')
        if 'pink' in ID:
            list_ID.append('pink')
        else:
            list_ID.append('white')
    else:
        targets = pd.read_excel(rf'{ID[:6]}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')
        list_ID.append('static')
    # Extract Excel file and convert them into real data
    target_signal_x, target_signal_y = lbs.convert_excel_to_screen_size_targets(targets, old_data)


    # Extract the data during game
    data = lbs.values_during_game(data)
    print(data.columns)



    # Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
    list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data, old_data)


    # Create a simple graph with all the columns you need to plot
    # lbs.simple_graph(list_with_all_df_separated_by_set, 'pitch', 'yaw', 'roll')

    # Create a scatter with a slider for visualization of target position vs player position
    # lbs.graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set)



    # Seperation of the x and y position of the player for each set
    df_5_list = lbs.big_list_to_5df_list(list_with_all_df_separated_by_set)
    position_player_x_set_1 = df_5_list[0]['player_pos_x'].to_numpy()
    position_player_x_set_2 = df_5_list[1]['player_pos_x'].to_numpy()
    position_player_x_set_3 = df_5_list[2]['player_pos_x'].to_numpy()
    position_player_x_set_4 = df_5_list[3]['player_pos_x'].to_numpy()
    position_player_x_set_5 = df_5_list[4]['player_pos_x'].to_numpy()
    position_player_y_set_1 = df_5_list[0]['player_pos_y'].to_numpy()
    position_player_y_set_2 = df_5_list[1]['player_pos_y'].to_numpy()
    position_player_y_set_3 = df_5_list[2]['player_pos_y'].to_numpy()
    position_player_y_set_4 = df_5_list[3]['player_pos_y'].to_numpy()
    position_player_y_set_5 = df_5_list[4]['player_pos_y'].to_numpy()



    # Assessment of power spectrum of position_player
    # power_90_set_1_x, power_95_set_1_x, power_99_set_1_x = lib.FFT(position_player_x_set_1, 40)
    # power_90_set_1_y, power_95_set_1_y, power_99_set_1_y = lib.FFT(position_player_y_set_1, 40)
    # power_90_set_2_x, power_95_set_2_x, power_99_set_2_x = lib.FFT(position_player_x_set_2, 40)
    # power_90_set_2_y, power_95_set_2_y, power_99_set_2_y = lib.FFT(position_player_y_set_2, 40)
    # power_90_set_3_x, power_95_set_3_x, power_99_set_3_x = lib.FFT(position_player_x_set_3, 40)
    # power_90_set_3_y, power_95_set_3_y, power_99_set_3_y = lib.FFT(position_player_y_set_3, 40)
    # power_90_set_4_x, power_95_set_4_x, power_99_set_4_x = lib.FFT(position_player_x_set_4, 40)
    # power_90_set_4_y, power_95_set_4_y, power_99_set_4_y = lib.FFT(position_player_y_set_4, 40)
    # power_90_set_5_x, power_95_set_5_x, power_99_set_5_x = lib.FFT(position_player_x_set_5, 40)
    # power_90_set_5_y, power_95_set_5_y, power_99_set_5_y = lib.FFT(position_player_y_set_5, 40)
    #
    # list_power_90_set_1_x.append(power_90_set_1_x)
    # list_power_90_set_2_x.append(power_90_set_2_x)
    # list_power_90_set_3_x.append(power_90_set_3_x)
    # list_power_90_set_4_x.append(power_90_set_4_x)
    # list_power_90_set_5_x.append(power_90_set_5_x)
    # list_power_90_set_1_y.append(power_90_set_1_y)
    # list_power_90_set_2_y.append(power_90_set_2_y)
    # list_power_90_set_3_y.append(power_90_set_3_y)
    # list_power_90_set_4_y.append(power_90_set_4_y)
    # list_power_90_set_5_y.append(power_90_set_5_y)
    # list_power_95_set_1_x.append(power_95_set_1_x)
    # list_power_95_set_2_x.append(power_95_set_2_x)
    # list_power_95_set_3_x.append(power_95_set_3_x)
    # list_power_95_set_4_x.append(power_95_set_4_x)
    # list_power_95_set_5_x.append(power_95_set_5_x)
    # list_power_95_set_1_y.append(power_95_set_1_y)
    # list_power_95_set_2_y.append(power_95_set_2_y)
    # list_power_95_set_3_y.append(power_95_set_3_y)
    # list_power_95_set_4_y.append(power_95_set_4_y)
    # list_power_95_set_5_y.append(power_95_set_5_y)
    # list_power_99_set_1_x.append(power_99_set_1_x)
    # list_power_99_set_2_x.append(power_99_set_2_x)
    # list_power_99_set_3_x.append(power_99_set_3_x)
    # list_power_99_set_4_x.append(power_99_set_4_x)
    # list_power_99_set_5_x.append(power_99_set_5_x)
    # list_power_99_set_1_y.append(power_99_set_1_y)
    # list_power_99_set_2_y.append(power_99_set_2_y)
    # list_power_99_set_3_y.append(power_99_set_3_y)
    # list_power_99_set_4_y.append(power_99_set_4_y)
    # list_power_99_set_5_y.append(power_99_set_5_y)

    # Apply Butterworth low pass at 5Hz
    # position_player_x_set_1 = lib.Butterworth(40, 5, position_player_x_set_1)
    # position_player_x_set_2 = lib.Butterworth(40, 5, position_player_x_set_2)
    # position_player_x_set_3 = lib.Butterworth(40, 5, position_player_x_set_3)
    # position_player_x_set_4 = lib.Butterworth(40, 5, position_player_x_set_4)
    # position_player_x_set_5 = lib.Butterworth(40, 5, position_player_x_set_5)
    # position_player_y_set_1 = lib.Butterworth(40, 5, position_player_y_set_1)
    # position_player_y_set_2 = lib.Butterworth(40, 5, position_player_y_set_2)
    # position_player_y_set_3 = lib.Butterworth(40, 5, position_player_y_set_3)
    # position_player_y_set_4 = lib.Butterworth(40, 5, position_player_y_set_4)
    # position_player_y_set_5 = lib.Butterworth(40, 5, position_player_y_set_5)

    # Calculate the travel distance for each set
    travel_distance_set_1 = lbs.travel_distance(position_player_x_set_1, position_player_y_set_1)
    travel_distance_set_2 = lbs.travel_distance(position_player_x_set_2, position_player_y_set_2)
    travel_distance_set_3 = lbs.travel_distance(position_player_x_set_3, position_player_y_set_3)
    travel_distance_set_4 = lbs.travel_distance(position_player_x_set_4, position_player_y_set_4)
    travel_distance_set_5 = lbs.travel_distance(position_player_x_set_5, position_player_y_set_5)


    # Calculation of SaEn per set per axis
    # SaEn_position_player_x_set_1 = lbs.Ent_Samp(position_player_x_set_1, 2, 0.2)
    # SaEn_position_player_x_set_2 = lbs.Ent_Samp(position_player_x_set_2, 2, 0.2)
    # SaEn_position_player_x_set_3 = lbs.Ent_Samp(position_player_x_set_3, 2, 0.2)
    # SaEn_position_player_x_set_4 = lbs.Ent_Samp(position_player_x_set_4, 2, 0.2)
    # SaEn_position_player_x_set_5 = lbs.Ent_Samp(position_player_x_set_5, 2, 0.2)
    # SaEn_position_player_y_set_1 = lbs.Ent_Samp(position_player_y_set_1, 2, 0.2)
    # SaEn_position_player_y_set_2 = lbs.Ent_Samp(position_player_y_set_2, 2, 0.2)
    # SaEn_position_player_y_set_3 = lbs.Ent_Samp(position_player_y_set_3, 2, 0.2)
    # SaEn_position_player_y_set_4 = lbs.Ent_Samp(position_player_y_set_4, 2, 0.2)
    # SaEn_position_player_y_set_5 = lbs.Ent_Samp(position_player_y_set_5, 2, 0.2)
    #
    # list_SaEn_x_set_1.append(SaEn_position_player_x_set_1)
    # list_SaEn_x_set_2.append(SaEn_position_player_x_set_2)
    # list_SaEn_x_set_3.append(SaEn_position_player_x_set_3)
    # list_SaEn_x_set_4.append(SaEn_position_player_x_set_4)
    # list_SaEn_x_set_5.append(SaEn_position_player_x_set_5)
    # list_SaEn_y_set_1.append(SaEn_position_player_y_set_1)
    # list_SaEn_y_set_2.append(SaEn_position_player_y_set_2)
    # list_SaEn_y_set_3.append(SaEn_position_player_y_set_3)
    # list_SaEn_y_set_4.append(SaEn_position_player_y_set_4)
    # list_SaEn_y_set_5.append(SaEn_position_player_y_set_5)
    #
    # # Calculation of SaEn per set of travel distance
    # SaEn_travel_distance_set_1 = lbs.Ent_Samp(travel_distance_set_1, 2, 0.2)
    # SaEn_travel_distance_set_2 = lbs.Ent_Samp(travel_distance_set_2, 2, 0.2)
    # SaEn_travel_distance_set_3 = lbs.Ent_Samp(travel_distance_set_3, 2, 0.2)
    # SaEn_travel_distance_set_4 = lbs.Ent_Samp(travel_distance_set_4, 2, 0.2)
    # SaEn_travel_distance_set_5 = lbs.Ent_Samp(travel_distance_set_5, 2, 0.2)
    #
    # list_SaEn_travel_distance_set_1.append(SaEn_travel_distance_set_1)
    # list_SaEn_travel_distance_set_2.append(SaEn_travel_distance_set_2)
    # list_SaEn_travel_distance_set_3.append(SaEn_travel_distance_set_3)
    # list_SaEn_travel_distance_set_4.append(SaEn_travel_distance_set_4)
    # list_SaEn_travel_distance_set_5.append(SaEn_travel_distance_set_5)
    #
    # # Calculation of DFA per set per axis
    # scales_x_set_1 = np.arange(4, len(position_player_x_set_1)//4)
    # scales_x_set_2 = np.arange(4, len(position_player_x_set_2)//4)
    # scales_x_set_3 = np.arange(4, len(position_player_x_set_3)//4)
    # scales_x_set_4 = np.arange(4, len(position_player_x_set_4)//4)
    # scales_x_set_5 = np.arange(4, len(position_player_x_set_5)//4)
    # scales_y_set_1 = np.arange(4, len(position_player_y_set_1)//4)
    # scales_y_set_2 = np.arange(4, len(position_player_y_set_2)//4)
    # scales_y_set_3 = np.arange(4, len(position_player_y_set_3)//4)
    # scales_y_set_4 = np.arange(4, len(position_player_y_set_4)//4)
    # scales_y_set_5 = np.arange(4, len(position_player_y_set_5)//4)
    #
    # _, _, DFA_x_set_1 = lbs.DFA_NONAN(position_player_x_set_1, scales_x_set_1, order=1, plot=False)
    # _, _, DFA_x_set_2 = lbs.DFA_NONAN(position_player_x_set_2, scales_x_set_2, order=1, plot=False)
    # _, _, DFA_x_set_3 = lbs.DFA_NONAN(position_player_x_set_3, scales_x_set_3, order=1, plot=False)
    # _, _, DFA_x_set_4 = lbs.DFA_NONAN(position_player_x_set_4, scales_x_set_4, order=1, plot=False)
    # _, _, DFA_x_set_5 = lbs.DFA_NONAN(position_player_x_set_5, scales_x_set_5, order=1, plot=False)
    # _, _, DFA_y_set_1 = lbs.DFA_NONAN(position_player_y_set_1, scales_y_set_1, order=1, plot=False)
    # _, _, DFA_y_set_2 = lbs.DFA_NONAN(position_player_y_set_2, scales_y_set_2, order=1, plot=False)
    # _, _, DFA_y_set_3 = lbs.DFA_NONAN(position_player_y_set_3, scales_y_set_3, order=1, plot=False)
    # _, _, DFA_y_set_4 = lbs.DFA_NONAN(position_player_y_set_4, scales_y_set_4, order=1, plot=False)
    # _, _, DFA_y_set_5 = lbs.DFA_NONAN(position_player_y_set_5, scales_y_set_5, order=1, plot=False)
    #
    # list_DFA_x_set_1.append(DFA_x_set_1)
    # list_DFA_x_set_2.append(DFA_x_set_2)
    # list_DFA_x_set_3.append(DFA_x_set_3)
    # list_DFA_x_set_4.append(DFA_x_set_4)
    # list_DFA_x_set_5.append(DFA_x_set_5)
    # list_DFA_y_set_1.append(DFA_y_set_1)
    # list_DFA_y_set_2.append(DFA_y_set_2)
    # list_DFA_y_set_3.append(DFA_y_set_3)
    # list_DFA_y_set_4.append(DFA_y_set_4)
    # list_DFA_y_set_5.append(DFA_y_set_5)
    #
    # # Calculation of DFA of travel distance
    # scales_set_1 = np.arange(4, len(travel_distance_set_1)//4)
    # scales_set_2 = np.arange(4, len(travel_distance_set_2)//4)
    # scales_set_3 = np.arange(4, len(travel_distance_set_3)//4)
    # scales_set_4 = np.arange(4, len(travel_distance_set_4)//4)
    # scales_set_5 = np.arange(4, len(travel_distance_set_5)//4)
    #
    # _, _, DFA_set_1 = lbs.DFA_NONAN(travel_distance_set_1, scales_set_1, order=1, plot=False)
    # _, _, DFA_set_2 = lbs.DFA_NONAN(travel_distance_set_2, scales_set_2, order=1, plot=False)
    # _, _, DFA_set_3 = lbs.DFA_NONAN(travel_distance_set_3, scales_set_3, order=1, plot=False)
    # _, _, DFA_set_4 = lbs.DFA_NONAN(travel_distance_set_4, scales_set_4, order=1, plot=False)
    # _, _, DFA_set_5 = lbs.DFA_NONAN(travel_distance_set_5, scales_set_5, order=1, plot=False)
    #
    # list_DFA_travel_distance_set_1.append(DFA_set_1)
    # list_DFA_travel_distance_set_2.append(DFA_set_2)
    # list_DFA_travel_distance_set_3.append(DFA_set_3)
    # list_DFA_travel_distance_set_4.append(DFA_set_4)
    # list_DFA_travel_distance_set_5.append(DFA_set_5)


    # Calculate and show the spatial error for each target
    spatial_error, list_time_stamp_of_min_spatial_error_separated_by_set = lbs.spatial_error_best_window(list_with_all_df_separated_by_set, plot=False, time_window=500)

    spatial_error_set_1_average = np.mean(spatial_error[0])
    spatial_error_set_2_average = np.mean(spatial_error[1])
    spatial_error_set_3_average = np.mean(spatial_error[2])
    spatial_error_set_4_average = np.mean(spatial_error[3])
    spatial_error_set_5_average = np.mean(spatial_error[4])
    spatial_error_set_1_sd = np.std(spatial_error[0])
    spatial_error_set_2_sd = np.std(spatial_error[1])
    spatial_error_set_3_sd = np.std(spatial_error[2])
    spatial_error_set_4_sd = np.std(spatial_error[3])
    spatial_error_set_5_sd = np.std(spatial_error[4])


    list_spatial_error_set_1_average.append(spatial_error_set_1_average)
    list_spatial_error_set_2_average.append(spatial_error_set_2_average)
    list_spatial_error_set_3_average.append(spatial_error_set_3_average)
    list_spatial_error_set_4_average.append(spatial_error_set_4_average)
    list_spatial_error_set_5_average.append(spatial_error_set_5_average)
    list_spatial_error_set_1_sd.append(spatial_error_set_1_sd)
    list_spatial_error_set_2_sd.append(spatial_error_set_2_sd)
    list_spatial_error_set_3_sd.append(spatial_error_set_3_sd)
    list_spatial_error_set_4_sd.append(spatial_error_set_4_sd)
    list_spatial_error_set_5_sd.append(spatial_error_set_5_sd)

#
#
# Calculate slopes, error, Average Spatial error, and sd spatial error at 500 for everyone
dist = {'ID': list_ID,
        'Exact ID': list_exact_ID,
        'Average Spatial error set 1': list_spatial_error_set_1_average,
        'Average Spatial error set 2': list_spatial_error_set_2_average,
        'Average Spatial error set 3': list_spatial_error_set_3_average,
        'Average Spatial error set 4': list_spatial_error_set_4_average,
        'Average Spatial error set 5': list_spatial_error_set_5_average,
        'Sd Spatial error set 1': list_spatial_error_set_1_sd,
        'Sd Spatial error set 2': list_spatial_error_set_2_sd,
        'Sd Spatial error set 3': list_spatial_error_set_3_sd,
        'Sd Spatial error set 4': list_spatial_error_set_4_sd,
        'Sd Spatial error set 5': list_spatial_error_set_5_sd,
        }

# dist_learning_analysis = {'ID': list_ID,
#                             'Exact ID': list_exact_ID,
#                             'SaEn_x_set_1': list_SaEn_x_set_1,
#                             'SaEn_x_set_2': list_SaEn_x_set_2,
#                             'SaEn_x_set_3': list_SaEn_x_set_3,
#                             'SaEn_x_set_4': list_SaEn_x_set_4,
#                             'SaEn_x_set_5': list_SaEn_x_set_5,
#                             'SaEn_y_set_1': list_SaEn_y_set_1,
#                             'SaEn_y_set_2': list_SaEn_y_set_2,
#                             'SaEn_y_set_3': list_SaEn_y_set_3,
#                             'SaEn_y_set_4': list_SaEn_y_set_4,
#                             'SaEn_y_set_5': list_SaEn_y_set_5,
#                             'SaEn_travel_distance_set_1': list_SaEn_travel_distance_set_1,
#                             'SaEn_travel_distance_set_2': list_SaEn_travel_distance_set_2,
#                             'SaEn_travel_distance_set_3': list_SaEn_travel_distance_set_3,
#                             'SaEn_travel_distance_set_4': list_SaEn_travel_distance_set_4,
#                             'SaEn_travel_distance_set_5': list_SaEn_travel_distance_set_5,
#                             'DFA_x_set_1': list_DFA_x_set_1,
#                             'DFA_x_set_2': list_DFA_x_set_2,
#                             'DFA_x_set_3': list_DFA_x_set_3,
#                             'DFA_x_set_4': list_DFA_x_set_4,
#                             'DFA_x_set_5': list_DFA_x_set_5,
#                             'DFA_y_set_1': list_DFA_y_set_1,
#                             'DFA_y_set_2': list_DFA_y_set_2,
#                             'DFA_y_set_3': list_DFA_y_set_3,
#                             'DFA_y_set_4': list_DFA_y_set_4,
#                             'DFA_y_set_5': list_DFA_y_set_5,
#                             'DFA_travel_distance_set_1': list_DFA_travel_distance_set_1,
#                             'DFA_travel_distance_set_2': list_DFA_travel_distance_set_2,
#                             'DFA_travel_distance_set_3': list_DFA_travel_distance_set_3,
#                             'DFA_travel_distance_set_4': list_DFA_travel_distance_set_4,
#                             'DFA_travel_distance_set_5': list_DFA_travel_distance_set_5,
#                           }



df_learning = pd.DataFrame(dist)


directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Results'
os.chdir(directory)
df_learning.to_excel('Results spatial error.xlsx')
for column in df_learning.columns:
    print(df_learning[column])
