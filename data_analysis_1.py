import lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

pd.set_option("display.max_rows", None)

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 3\Data'
os.chdir(directory)

data = pd.read_csv(r'record_173754625923.txt')

target_pos_x, target_pos_y, player_pos_x, player_pos_y = lbs.values_during_game(data)

spatial_error_all = lbs.spatial_error_calculation(target_pos_x, target_pos_y, player_pos_x, player_pos_y)

target_pos_x, target_pos_y, player_pos_x, player_pos_y = lbs.return_the_values_before_target_change(target_pos_x, target_pos_y, player_pos_x, player_pos_y)

spatial_error_30 = lbs.spatial_error_calculation(target_pos_x, target_pos_y, player_pos_x, player_pos_y)

# print(game_data.columns)
# print(type(game_data['target_pos_y'][25928]))
plt.scatter(target_pos_x, target_pos_y, label='target')
plt.scatter(player_pos_x, player_pos_y, label='player')
plt.legend()
plt.show()

plt.plot(spatial_error_all)
plt.show()

plt.plot(spatial_error_30)
plt.show()



