import Lib_squats as lbs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

import lib

pink_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data\Valid Data\pink1'
os.chdir(pink_directory)
pink = pd.read_csv(r'pink1.txt')

static_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data\Valid Data\static1'
os.chdir(static_directory)
static = pd.read_csv(r'static1.txt')

white_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data\Valid Data\white1'
os.chdir(white_directory)
white = pd.read_csv(r'white1.txt')

data_dfs = [pink, static, white]
names = ['pink22', 'static29', 'white29']

list_signals = []
list_times = []
for name, data in zip(names, data_dfs):


    data = lbs.values_during_game(data)
    print(data.columns)
    list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data, False)

    time_of_target_appearance = []
    for i in range(len(list_with_all_df_separated_by_set)):
        for j in range(len(list_with_all_df_separated_by_set[0])):
            first_appearance = list_with_all_df_separated_by_set[0][0]['timestamp'].iloc[0]
            last_appearance = list_with_all_df_separated_by_set[0][0]['timestamp'].iloc[-1]
            difference = last_appearance - first_appearance
            # make it seconds
            difference = difference / 1000
            time_of_target_appearance.append(difference)


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

    # Calculate the travel distance for each set
    travel_distance_set_1 = lbs.travel_distance(position_player_x_set_1, position_player_y_set_1)
    travel_distance_set_2 = lbs.travel_distance(position_player_x_set_2, position_player_y_set_2)
    travel_distance_set_3 = lbs.travel_distance(position_player_x_set_3, position_player_y_set_3)
    travel_distance_set_4 = lbs.travel_distance(position_player_x_set_4, position_player_y_set_4)
    travel_distance_set_5 = lbs.travel_distance(position_player_x_set_5, position_player_y_set_5)
    time = np.linspace(0, len(travel_distance_set_5) / 40, len(travel_distance_set_5))

    list_signals.append(travel_distance_set_5)
    list_times.append(time)
print(len(list_times[0]))
print(len(list_times[1]))
print(len(list_times[2]))

def create_sine_wave(
    magnitude=1.0,     # amplitude
    cycles=1,          # number of cycles
    duration=1.0,      # seconds
    fs=1000            # sampling frequency (Hz)
):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    y = magnitude * np.sin(2 * np.pi * cycles * t / duration)
    return t, y

# t, y = create_sine_wave(magnitude=50, cycles=22, duration=5, fs=40)
# y = y -min(y)
# print(len(y))


scales = np.arange(4, len(list_signals[0])//4)

lib.DFA_NONAN(list_signals[0], scales, plot=True)
lib.DFA_NONAN(list_signals[1], scales, plot=True)
lib.DFA_NONAN(list_signals[2], scales, plot=True)


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list_times[0][:2600],
        y=list_signals[0][:2600],
        mode="lines",
        name="Pink"
    )
)

fig.add_trace(
    go.Scatter(
        x=list_times[1][:2600],
        y=list_signals[1][:2600],
        mode="lines",
        name="Static"
    )
)

# fig.add_trace(
#     go.Scatter(
#         x=list_times[2][:2600],
#         y=list_signals[2][:2600],
#         mode="lines",
#         name="White"
#     )
# )

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Signal",
    legend_title="Signal type",
    template="simple_white"
)

# fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Sine wave"))
# fig.update_layout(
#     xaxis_title="Time (s)",
#     yaxis_title="Amplitude",
#     template="simple_white"
# )

fig.show()


