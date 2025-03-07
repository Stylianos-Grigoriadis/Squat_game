import Lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob
import seaborn as sns


pd.set_option("display.max_rows", None)

participants_before_change = ['pink1', 'pink10', 'pink11', 'pink12', 'pink13', 'pink14','pink15', 'pink2', 'pink3', 'pink4', 'pink5', 'pink6', 'pink7', 'pink8', 'pink9', 'static1', 'static10', 'static11', 'static12', 'static13', 'static2', 'static3', 'static4', 'static5', 'static6', 'static7', 'static8', 'static9', 'white1', 'white10', 'white11', 'white12', 'white13', 'white14', 'white2', 'white3', 'white4', 'white5', 'white6', 'white7', 'white8', 'white9']
participants_after_change = []


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Data'
files = glob.glob(os.path.join(directory_path, "*"))
list_RMSE = []
list_slope = []
list_ID = []
list_average_spatial_error = []
list_sd_spatial_error = []


for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
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


    # Extract the
    data = lbs.values_during_game(data)
    # print(data.columns)


    # Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
    list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data, old_data)
    # print(list_with_all_df_separated_by_set[3])



    # Comparison of Excel signal and the generated signal
    # if old_data == False:
    #     target_pos_x_list = []
    #     target_pos_y_list = []
    #
    #     for list in list_with_all_df_separated_by_set:
    #         for df in list:
    #             target_pos_x_list.append(df['target_pos_x'][0])
    #             target_pos_y_list.append(df['target_pos_y'][0])
    #     for i in range(len(target_pos_x_list)):
    #         if target_pos_x_list[i] < 128:
    #             print('aaaaaaaaaaaaaaaaaaa')
    #             target_pos_x_list[i]= 128
    #         if target_pos_x_list[i] > 1980 - 128:
    #             print('aaaaaaaaaaaaaaaaaaa')
    #
    #             target_pos_x_list[i] = 1980 - 128
    #
    #     for i in range(len(target_pos_y_list)):
    #         if target_pos_y_list[i] < 128:
    #             print('aaaaaaaaaaaaaaaaaaa')
    #
    #             target_pos_y_list[i]= 128
    #         if target_pos_y_list[i] > 1080 - 128:
    #             print('aaaaaaaaaaaaaaaaaaa')
    #
    #             target_pos_y_list[i] = 1080 - 128
    #
    #
    #     plt.scatter(target_pos_x_list, target_pos_y_list, label='Generated')
    #     plt.scatter(target_signal_x, target_signal_y, label='Excel')
    #     plt.legend()
    #     plt.show()
    #
    #     plt.plot(target_pos_x_list, label='Generated')
    #     plt.plot(target_signal_x, label='Excel')
    #     plt.legend()
    #     plt.show()





    # Create a simple graph with all the columns you need to plot
    # lbs.simple_graph(list_with_all_df_separated_by_set, 'pitch', 'yaw', 'roll')

    # Create a scatter with a slider for visualization of target position vs player position
    # lbs.graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set)

    # Calculate and print the spatial error for each target
    # plt.title(f'{ID} 500')
    spatial_error_500 = lbs.spatial_error_best_window(list_with_all_df_separated_by_set, plot=False, time_window=500)
    # plt.title(f'{ID} 100')
    # spatial_error_300 = lbs.spatial_error_best_window(list_with_all_df_separated_by_set, plot=False, time_window=300)

    spatial_error_500, indeces_500 = lbs.list_of_five_list_flatten_list(spatial_error_500)
    # spatial_error_300, indeces_300 = lbs.list_of_five_list_flatten_list(spatial_error_300)


    time = np.linspace(0, len(spatial_error_500), len(spatial_error_500))
    plt.scatter(time, spatial_error_500, c='red', marker='x', label='500')
    # # plt.scatter(time, spatial_error_300, c='blue', marker='o', label='300')

    slope_500, intercept_500 = np.polyfit(time, spatial_error_500, 1)
    # slope_300, intercept_300 = np.polyfit(time, spatial_error_300, 1)

    fit_500 = slope_500 * time + intercept_500
    # fit_300 = slope_300 * time + intercept_300

    residuals = spatial_error_500 - fit_500
    MSE = np.mean(residuals**2)
    RMSE = np.sqrt(MSE)

    list_slope.append(slope_500)
    list_RMSE.append(RMSE)

    average = np.mean(spatial_error_500)
    list_average_spatial_error.append(average)
    sd = np.std(spatial_error_500)
    list_sd_spatial_error.append(sd)



    plt.plot(time, fit_500, c='red', label="Best Fit Line 500")
    # plt.plot(time, fit_300, c='blue', label="Best Fit Line 300")

    for i in indeces_500:
        plt.axvline(x=i, linestyle='--', c='k')
    plt.legend()
    plt.ylim(0, 800)
    # plt.title(f'{ID}\nslope 500 = {slope_500}\nslope 100 = {slope_300}')
    plt.show()

# Calculate slopes and error at 500 for everyone
dist = {'ID': list_ID,
        'slope': list_slope,
        'RMSE': list_RMSE,
        'Average': list_average_spatial_error,
        'Sd': list_sd_spatial_error
        }
df = pd.DataFrame(dist)
directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Results'
os.chdir(directory)
df.to_excel('Slope.xlsx')

# Create a box plot for the 'slope' based on 'ID'
plt.figure(figsize=(12, 6))

# Box plot for slope
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='ID', y='slope', color='lightblue')
plt.title('Box Plot of Slope by ID')
plt.xlabel('ID')
plt.ylabel('Slope')

# Box plot for RMSE
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='ID', y='RMSE', color='lightgreen')
plt.title('Box Plot of RMSE by ID')
plt.xlabel('ID')
plt.ylabel('RMSE')

# Show the plot
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Box plot for slope
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='ID', y='Average', color='lightblue')
plt.title('Box Plot of Average by ID')
plt.xlabel('ID')
plt.ylabel('Slope')

# Box plot for RMSE
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='ID', y='Sd', color='lightgreen')
plt.title('Box Plot of Sd by ID')
plt.xlabel('ID')
plt.ylabel('RMSE')

# Show the plot
plt.tight_layout()
plt.show()
