import lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

pd.set_option("display.max_rows", None)

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 3\Data'
os.chdir(directory)

data = pd.read_csv(r'record_173754625923.txt')

game_data = lbs.values_during_game(data)
print(game_data)
# print(game_data.columns)
# print(type(game_data['target_pos_y'][25928]))
# plt.scatter(game_data['target_pos_x'], game_data['target_pos_y'])
# plt.show()