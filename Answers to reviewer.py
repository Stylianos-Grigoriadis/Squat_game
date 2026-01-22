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

sample_freq = [40, 20, 10]
for freq in sample_freq:


    list_ID = []
    list_exact_ID = []

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
        # Extract the data during game
        data = lbs.values_during_game(data)
        print(data.columns)



        # Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
        list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data, old_data)

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

        if freq == 40:
            position_player_x_set_1 = position_player_x_set_1
            position_player_x_set_2 = position_player_x_set_2
            position_player_x_set_3 = position_player_x_set_3
            position_player_x_set_4 = position_player_x_set_4
            position_player_x_set_5 = position_player_x_set_5
            position_player_y_set_1 = position_player_y_set_1
            position_player_y_set_2 = position_player_y_set_2
            position_player_y_set_3 = position_player_y_set_3
            position_player_y_set_4 = position_player_y_set_4
            position_player_y_set_5 = position_player_y_set_5
        else:
            position_player_x_set_1 = lib.resample_signal(position_player_x_set_1, 40, freq, anti_aliasing=True)
            position_player_x_set_2 = lib.resample_signal(position_player_x_set_2, 40, freq, anti_aliasing=True)
            position_player_x_set_3 = lib.resample_signal(position_player_x_set_3, 40, freq, anti_aliasing=True)
            position_player_x_set_4 = lib.resample_signal(position_player_x_set_4, 40, freq, anti_aliasing=True)
            position_player_x_set_5 = lib.resample_signal(position_player_x_set_5, 40, freq, anti_aliasing=True)
            position_player_y_set_1 = lib.resample_signal(position_player_y_set_1, 40, freq, anti_aliasing=True)
            position_player_y_set_2 = lib.resample_signal(position_player_y_set_2, 40, freq, anti_aliasing=True)
            position_player_y_set_3 = lib.resample_signal(position_player_y_set_3, 40, freq, anti_aliasing=True)
            position_player_y_set_4 = lib.resample_signal(position_player_y_set_4, 40, freq, anti_aliasing=True)
            position_player_y_set_5 = lib.resample_signal(position_player_y_set_5, 40, freq, anti_aliasing=True)

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

        # Calculation of DFA of travel distance
        scales_set_1 = np.arange(4, len(travel_distance_set_1)//4)
        scales_set_2 = np.arange(4, len(travel_distance_set_2)//4)
        scales_set_3 = np.arange(4, len(travel_distance_set_3)//4)
        scales_set_4 = np.arange(4, len(travel_distance_set_4)//4)
        scales_set_5 = np.arange(4, len(travel_distance_set_5)//4)

        _, _, DFA_set_1 = lbs.DFA_NONAN(travel_distance_set_1, scales_set_1, order=1, plot=False)
        _, _, DFA_set_2 = lbs.DFA_NONAN(travel_distance_set_2, scales_set_2, order=1, plot=False)
        _, _, DFA_set_3 = lbs.DFA_NONAN(travel_distance_set_3, scales_set_3, order=1, plot=False)
        _, _, DFA_set_4 = lbs.DFA_NONAN(travel_distance_set_4, scales_set_4, order=1, plot=False)
        _, _, DFA_set_5 = lbs.DFA_NONAN(travel_distance_set_5, scales_set_5, order=1, plot=False)

        list_DFA_travel_distance_set_1.append(DFA_set_1)
        list_DFA_travel_distance_set_2.append(DFA_set_2)
        list_DFA_travel_distance_set_3.append(DFA_set_3)
        list_DFA_travel_distance_set_4.append(DFA_set_4)
        list_DFA_travel_distance_set_5.append(DFA_set_5)


    dist_learning_analysis = {'ID': list_ID,
                                'Exact ID': list_exact_ID,
                                'DFA_travel_distance_set_1': list_DFA_travel_distance_set_1,
                                'DFA_travel_distance_set_2': list_DFA_travel_distance_set_2,
                                'DFA_travel_distance_set_3': list_DFA_travel_distance_set_3,
                                'DFA_travel_distance_set_4': list_DFA_travel_distance_set_4,
                                'DFA_travel_distance_set_5': list_DFA_travel_distance_set_5,
                              }



    df_learning = pd.DataFrame(dist_learning_analysis)

    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Writting\Publication tryouts\Journal of Sports Sciences\Review 1'
    os.chdir(directory)
    df_learning.to_excel(rf'Results the old way fc {freq}.xlsx')
