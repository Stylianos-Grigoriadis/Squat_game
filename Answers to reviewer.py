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




list_ID = []
list_exact_ID = []

list_weight_L_set_1_99 = []
list_weight_L_set_2_99 = []
list_weight_L_set_3_99 = []
list_weight_L_set_4_99 = []
list_weight_L_set_5_99 = []
list_weight_R_set_1_99 = []
list_weight_R_set_2_99 = []
list_weight_R_set_3_99 = []
list_weight_R_set_4_99 = []
list_weight_R_set_5_99 = []
list_pitch_set_1_99 = []
list_pitch_set_2_99 = []
list_pitch_set_3_99 = []
list_pitch_set_4_99 = []
list_pitch_set_5_99 = []
list_roll_set_1_99 = []
list_roll_set_2_99 = []
list_roll_set_3_99 = []
list_roll_set_4_99 = []
list_roll_set_5_99 = []
list_yaw_set_1_99 = []
list_yaw_set_2_99 = []
list_yaw_set_3_99 = []
list_yaw_set_4_99 = []
list_yaw_set_5_99 = []


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
    print(df_5_list[0].columns)
    weight_L_set_1 = df_5_list[0]['left_plate'].to_numpy()
    weight_L_set_2 = df_5_list[1]['left_plate'].to_numpy()
    weight_L_set_3 = df_5_list[2]['left_plate'].to_numpy()
    weight_L_set_4 = df_5_list[3]['left_plate'].to_numpy()
    weight_L_set_5 = df_5_list[4]['left_plate'].to_numpy()
    weight_R_set_1 = df_5_list[0]['right_plate'].to_numpy()
    weight_R_set_2 = df_5_list[1]['right_plate'].to_numpy()
    weight_R_set_3 = df_5_list[2]['right_plate'].to_numpy()
    weight_R_set_4 = df_5_list[3]['right_plate'].to_numpy()
    weight_R_set_5 = df_5_list[4]['right_plate'].to_numpy()
    pitch_set_1 = df_5_list[0]['pitch'].to_numpy()
    pitch_set_2 = df_5_list[1]['pitch'].to_numpy()
    pitch_set_3 = df_5_list[2]['pitch'].to_numpy()
    pitch_set_4 = df_5_list[3]['pitch'].to_numpy()
    pitch_set_5 = df_5_list[4]['pitch'].to_numpy()
    roll_set_1 = df_5_list[0]['roll'].to_numpy()
    roll_set_2 = df_5_list[1]['roll'].to_numpy()
    roll_set_3 = df_5_list[2]['roll'].to_numpy()
    roll_set_4 = df_5_list[3]['roll'].to_numpy()
    roll_set_5 = df_5_list[4]['roll'].to_numpy()
    yaw_set_1 = df_5_list[0]['yaw'].to_numpy()
    yaw_set_2 = df_5_list[1]['yaw'].to_numpy()
    yaw_set_3 = df_5_list[2]['yaw'].to_numpy()
    yaw_set_4 = df_5_list[3]['yaw'].to_numpy()
    yaw_set_5 = df_5_list[4]['yaw'].to_numpy()
    _, _, FFT_weight_L_set_1_99 = lib.FFT(weight_L_set_1, 40)
    _, _, FFT_weight_L_set_2_99 = lib.FFT(weight_L_set_2, 40)
    _, _, FFT_weight_L_set_3_99 = lib.FFT(weight_L_set_3, 40)
    _, _, FFT_weight_L_set_4_99 = lib.FFT(weight_L_set_4, 40)
    _, _, FFT_weight_L_set_5_99 = lib.FFT(weight_L_set_5, 40)
    _, _, FFT_weight_R_set_1_99 = lib.FFT(weight_R_set_1, 40)
    _, _, FFT_weight_R_set_2_99 = lib.FFT(weight_R_set_2, 40)
    _, _, FFT_weight_R_set_3_99 = lib.FFT(weight_R_set_3, 40)
    _, _, FFT_weight_R_set_4_99 = lib.FFT(weight_R_set_4, 40)
    _, _, FFT_weight_R_set_5_99 = lib.FFT(weight_R_set_5, 40)
    _, _, FFT_pitch_set_1_99 = lib.FFT(pitch_set_1, 40)
    _, _, FFT_pitch_set_2_99 = lib.FFT(pitch_set_2, 40)
    _, _, FFT_pitch_set_3_99 = lib.FFT(pitch_set_3, 40)
    _, _, FFT_pitch_set_4_99 = lib.FFT(pitch_set_4, 40)
    _, _, FFT_pitch_set_5_99 = lib.FFT(pitch_set_5, 40)
    _, _, FFT_roll_set_1_99 = lib.FFT(roll_set_1, 40)
    _, _, FFT_roll_set_2_99 = lib.FFT(roll_set_2, 40)
    _, _, FFT_roll_set_3_99 = lib.FFT(roll_set_3, 40)
    _, _, FFT_roll_set_4_99 = lib.FFT(roll_set_4, 40)
    _, _, FFT_roll_set_5_99 = lib.FFT(roll_set_5, 40)
    _, _, FFT_yaw_set_1_99 = lib.FFT(yaw_set_1, 40)
    _, _, FFT_yaw_set_2_99 = lib.FFT(yaw_set_2, 40)
    _, _, FFT_yaw_set_3_99 = lib.FFT(yaw_set_3, 40)
    _, _, FFT_yaw_set_4_99 = lib.FFT(yaw_set_4, 40)
    _, _, FFT_yaw_set_5_99 = lib.FFT(yaw_set_5, 40)

    list_weight_L_set_1_99.append(FFT_weight_L_set_1_99)
    list_weight_L_set_2_99.append(FFT_weight_L_set_2_99)
    list_weight_L_set_3_99.append(FFT_weight_L_set_3_99)
    list_weight_L_set_4_99.append(FFT_weight_L_set_4_99)
    list_weight_L_set_5_99.append(FFT_weight_L_set_5_99)
    list_weight_R_set_1_99.append(FFT_weight_R_set_1_99)
    list_weight_R_set_2_99.append(FFT_weight_R_set_2_99)
    list_weight_R_set_3_99.append(FFT_weight_R_set_3_99)
    list_weight_R_set_4_99.append(FFT_weight_R_set_4_99)
    list_weight_R_set_5_99.append(FFT_weight_R_set_5_99)
    list_pitch_set_1_99.append(FFT_pitch_set_1_99)
    list_pitch_set_2_99.append(FFT_pitch_set_2_99)
    list_pitch_set_3_99.append(FFT_pitch_set_3_99)
    list_pitch_set_4_99.append(FFT_pitch_set_4_99)
    list_pitch_set_5_99.append(FFT_pitch_set_5_99)
    list_roll_set_1_99.append(FFT_roll_set_1_99)
    list_roll_set_2_99.append(FFT_roll_set_2_99)
    list_roll_set_3_99.append(FFT_roll_set_3_99)
    list_roll_set_4_99.append(FFT_roll_set_4_99)
    list_roll_set_5_99.append(FFT_roll_set_5_99)
    list_yaw_set_1_99.append(FFT_yaw_set_1_99)
    list_yaw_set_2_99.append(FFT_yaw_set_2_99)
    list_yaw_set_3_99.append(FFT_yaw_set_3_99)
    list_yaw_set_4_99.append(FFT_yaw_set_4_99)
    list_yaw_set_5_99.append(FFT_yaw_set_5_99)

dist_learning_analysis = {'ID': list_ID,
                            'Exact ID': list_exact_ID,
                            'weight_L_set_1_99': list_weight_L_set_1_99,
                            'weight_L_set_2_99': list_weight_L_set_2_99,
                            'weight_L_set_3_99': list_weight_L_set_3_99,
                            'weight_L_set_4_99': list_weight_L_set_4_99,
                            'weight_L_set_5_99': list_weight_L_set_5_99,
                            'weight_R_set_1_99': list_weight_R_set_1_99,
                            'weight_R_set_2_99': list_weight_R_set_2_99,
                            'weight_R_set_3_99': list_weight_R_set_3_99,
                            'weight_R_set_4_99': list_weight_R_set_4_99,
                            'weight_R_set_5_99': list_weight_R_set_5_99,
                            'pitch_set_1_99': list_pitch_set_1_99,
                            'pitch_set_2_99': list_pitch_set_2_99,
                            'pitch_set_3_99': list_pitch_set_3_99,
                            'pitch_set_4_99': list_pitch_set_4_99,
                            'pitch_set_5_99': list_pitch_set_5_99,
                            'roll_set_1_99': list_roll_set_1_99,
                            'roll_set_2_99': list_roll_set_2_99,
                            'roll_set_3_99': list_roll_set_3_99,
                            'roll_set_4_99': list_roll_set_4_99,
                            'roll_set_5_99': list_roll_set_5_99,
                            'yaw_set_1_99': list_yaw_set_1_99,
                            'yaw_set_2_99': list_yaw_set_2_99,
                            'yaw_set_3_99': list_yaw_set_3_99,
                            'yaw_set_4_99': list_yaw_set_4_99,
                            'yaw_set_5_99': list_yaw_set_5_99,
                          }



df_learning = pd.DataFrame(dist_learning_analysis)

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Writting\Publication tryouts\Journal of Sports Sciences\Review 1'
os.chdir(directory)
df_learning.to_excel(rf'FFT_99.xlsx')
