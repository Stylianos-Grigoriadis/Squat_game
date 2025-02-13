import lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob


pd.set_option("display.max_rows", None)


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 6\Data'
files = glob.glob(os.path.join(directory_path, "*"))
for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    print(ID)
    if not ID[:-1] == 'static':
        targets = pd.read_excel(rf'{ID}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')
    else:
        targets = pd.read_excel(rf'{ID[:-1]}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')

    target_signal_x, target_signal_y = lbs.convert_excel_to_screen_size_targets(targets)
    target_pos_x, target_pos_y, player_pos_x, player_pos_y, left_plate, right_plate, pitch, yaw, roll = lbs.values_during_game(data)

    # target_pos_x, target_pos_y, player_pos_x, player_pos_y = lbs.return_the_values_before_target_change(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
    # lbs.graph_creation_target_vs_player(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
    #
    # lbs.graph_creation_of_spatial_error(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
    print(data.columns)
    left_plate = left_plate*(-1)
    both_plates = right_plate + left_plate

    plt.plot(yaw, linestyle='-', label='yaw')


    # plt.scatter(target_pos_x, target_pos_y, label='targets')
    plt.legend()
    plt.show()


    #
    # plt.plot(right_plate, label='right_plate')
    # plt.plot(left_plate, label='left_plate')
    # plt.legend()
    # plt.show()







