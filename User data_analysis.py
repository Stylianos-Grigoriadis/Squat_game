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



    # Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
    list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data, old_data)


    # Create a simple graph with all the columns you need to plot
    # lbs.simple_graph(list_with_all_df_separated_by_set, 'pitch', 'yaw', 'roll')

    # Create a scatter with a slider for visualization of target position vs player position
    # lbs.graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set)

    # Calculate and print the spatial error for each target
    spatial_error, list_time_stamp_of_min_spatial_error_separated_by_set = lbs.spatial_error_best_window(list_with_all_df_separated_by_set, plot=False, time_window=500)


    # Flatten the list with the spatial errors
    spatial_error, indices = lbs.list_of_five_lists_flatten_list(spatial_error)


    # Create the timestamps without space between sets
    differences_in_time_between_targets = []
    for time_list in list_time_stamp_of_min_spatial_error_separated_by_set:
        time_list = np.array(time_list)
        differences = np.diff(time_list)
        differences_in_time_between_targets.append(differences)
    differences_in_time_between_targets, _ = lbs.list_of_five_lists_flatten_list(differences_in_time_between_targets)
    differences_in_time_between_targets = np.array(differences_in_time_between_targets)
    average_difference = np.mean(differences_in_time_between_targets)
    time_stamps_without_between_set_space = []
    last_time_stamp = 0
    for time_list in list_time_stamp_of_min_spatial_error_separated_by_set:
        initial_time = time_list[0]
        for time_stamp in time_list:
            time_stamp = time_stamp - initial_time + last_time_stamp
            time_stamps_without_between_set_space.append(time_stamp)
        last_time_stamp = time_stamp + average_difference
    time_stamps_without_between_set_space = np.array(time_stamps_without_between_set_space)


    # Calculate the best fit line for the spatial error
    slope, intercept = np.polyfit(time_stamps_without_between_set_space, spatial_error, 1)
    fit = slope * time_stamps_without_between_set_space + intercept

    # Plot Spatial Error with the best linear fit line
    plt.scatter(time_stamps_without_between_set_space, spatial_error, c='red', marker='x', label='500')
    plt.plot(time_stamps_without_between_set_space, fit, c='red', label="Best Fit Line 500")
    set_time_stamps = []
    for i in [0, 29, 59, 89, 119, 149]:
        set_time_stamps.append(time_stamps_without_between_set_space[i])
    for i in set_time_stamps:
        plt.axvline(x=i, linestyle='--', c='k')
    plt.legend()
    plt.ylim(0, 800)
    plt.title(f'{ID}\nslope= {slope}')
    plt.ylabel('Spatial Error')
    plt.show()

    # # Calculate the segmented linear regression
    # estimate_target_where_the_change_would_happen = 75
    # targets_before_and_after = 70
    # low_limit = estimate_target_where_the_change_would_happen - targets_before_and_after
    # upper_limit = estimate_target_where_the_change_would_happen + targets_before_and_after
    # Average_residuals_before = []
    # Average_residuals_after = []
    # spatial_error_before_list = []
    # spatial_error_after_list = []
    # time_stamps_without_between_set_space_before_list = []
    # time_stamps_without_between_set_space_after_list = []
    # fit_before_list = []
    # fit_after_list = []
    # slope_before_list = []
    # slope_after_list = []
    # intercept_before_list = []
    # intercept_after_list = []
    #
    #
    # for i in range(low_limit, upper_limit + 1):
    #
    #     spatial_error_before = spatial_error[0:i]
    #     time_stamps_without_between_set_space_before = time_stamps_without_between_set_space[0:i]
    #     spatial_error_after = spatial_error[i:]
    #     time_stamps_without_between_set_space_after = time_stamps_without_between_set_space[i:]
    #
    #     slope_before, intercept_before = np.polyfit(time_stamps_without_between_set_space_before, spatial_error_before, 1)
    #     fit_before = slope_before * time_stamps_without_between_set_space_before + intercept_before
    #     slope_after, intercept_after = np.polyfit(time_stamps_without_between_set_space_after, spatial_error_after, 1)
    #     fit_after = slope_after * time_stamps_without_between_set_space_after + intercept_after
    #
    #     residuals_before = spatial_error_before - fit_before
    #     MSE_before = np.mean(residuals_before ** 2)
    #     RMSE_before = np.sqrt(MSE_before)
    #
    #     residuals_after = spatial_error_after - fit_after
    #     MSE_after = np.mean(residuals_after ** 2)
    #     RMSE_after = np.sqrt(MSE_after)
    #
    #     Average_residuals_before.append(RMSE_before)
    #     Average_residuals_after.append(RMSE_after)
    #     spatial_error_before_list.append(spatial_error_before)
    #     spatial_error_after_list.append(spatial_error_after)
    #     time_stamps_without_between_set_space_before_list.append(time_stamps_without_between_set_space_before)
    #     time_stamps_without_between_set_space_after_list.append(time_stamps_without_between_set_space_after)
    #     fit_before_list.append(fit_before)
    #     fit_after_list.append(fit_after)
    #     slope_before_list.append(slope_before)
    #     slope_after_list.append(slope_after)
    #     intercept_before_list.append(intercept_before)
    #     intercept_after_list.append(intercept_after)
    #
    # sum_Average_residuals_before_after = []
    #
    #
    # for i in range(len(Average_residuals_before)):
    #
    #     sum = Average_residuals_before[i]+Average_residuals_after[i]
    #     sum_Average_residuals_before_after.append(sum)
    #
    # index_min_sum_Average_residuals_before_after = sum_Average_residuals_before_after.index(min(sum_Average_residuals_before_after))
    #
    # best_Average_residuals_before = Average_residuals_before[index_min_sum_Average_residuals_before_after]
    # best_Average_residuals_after = Average_residuals_after[index_min_sum_Average_residuals_before_after]
    # best_spatial_error_before = spatial_error_before_list[index_min_sum_Average_residuals_before_after]
    # best_spatial_error_after = spatial_error_after_list[index_min_sum_Average_residuals_before_after]
    # best_time_stamps_without_between_set_space_before = time_stamps_without_between_set_space_before_list[index_min_sum_Average_residuals_before_after]
    # best_time_stamps_without_between_set_space_after = time_stamps_without_between_set_space_after_list[index_min_sum_Average_residuals_before_after]
    # best_fit_before = fit_before_list[index_min_sum_Average_residuals_before_after]
    # best_fit_after = fit_after_list[index_min_sum_Average_residuals_before_after]
    # best_slope_before = slope_before_list[index_min_sum_Average_residuals_before_after]
    # best_slope_after = slope_after_list[index_min_sum_Average_residuals_before_after]
    # best_intercept_before = intercept_before_list[index_min_sum_Average_residuals_before_after]
    # best_intercept_after = intercept_after_list[index_min_sum_Average_residuals_before_after]
    #
    # plt.plot(sum_Average_residuals_before_after)
    # plt.show()
    #
    # plt.scatter(best_time_stamps_without_between_set_space_before, best_spatial_error_before, c='red', marker='x', label='before')
    # plt.plot(best_time_stamps_without_between_set_space_before, best_fit_before, c='red', label="Best Fit Line before")
    # plt.scatter(best_time_stamps_without_between_set_space_after, best_spatial_error_after, c='blue', marker='x', label='after')
    # plt.plot(best_time_stamps_without_between_set_space_after, best_fit_after, c='blue', label="Best Fit Line after")
    # set_time_stamps = []
    # for i in [0, 29, 59, 89, 119, 149]:
    #     set_time_stamps.append(time_stamps_without_between_set_space[i])
    # for i in set_time_stamps:
    #     plt.axvline(x=i, linestyle='--', c='k')
    # plt.legend()
    # plt.ylim(0, 800)
    # plt.title(f'{ID}\nslope= {slope}')
    # plt.ylabel('Spatial Error')
    # plt.show()



    # # Calculate the segmented linear regression
    pw_fit = piecewise_regression.Fit(time_stamps_without_between_set_space, spatial_error, max_iterations=500, min_distance_to_edge=0.1, n_breakpoints=1, n_boot=1000)
    pw_fit.summary()

    # Plot the data, fit, breakpoints and confidence intervals
    pw_fit.plot_data(color="grey", s=20)
    # Pass in standard matplotlib keywords to control any of the plots
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.close()




    # Append values to create the df after with the results
    residuals = spatial_error - fit
    MSE = np.mean(residuals**2)
    RMSE = np.sqrt(MSE)

    list_slope.append(slope)
    list_RMSE.append(RMSE)
    average = np.mean(spatial_error)
    list_average_spatial_error.append(average)
    sd = np.std(spatial_error)
    list_sd_spatial_error.append(sd)

# Calculate slopes, error, Average Spatial error, and sd spatial error at 500 for everyone
dist = {'ID': list_ID,
        'slope': list_slope,
        'RMSE': list_RMSE,
        'Average': list_average_spatial_error,
        'Sd': list_sd_spatial_error
        }
df = pd.DataFrame(dist)
# directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Results'
# os.chdir(directory)
# df.to_excel('Slope.xlsx')
#
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
