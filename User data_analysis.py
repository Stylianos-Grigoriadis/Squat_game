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




pd.set_option("display.max_rows", None)

participants_before_change = ['pink1', 'pink10', 'pink11', 'pink12', 'pink13', 'pink14','pink15', 'pink2', 'pink3', 'pink4', 'pink5', 'pink6', 'pink7', 'pink8', 'pink9', 'static1', 'static10', 'static11', 'static12', 'static13', 'static2', 'static3', 'static4', 'static5', 'static6', 'static7', 'static8', 'static9', 'white1', 'white10', 'white11', 'white12', 'white13', 'white14', 'white2', 'white3', 'white4', 'white5', 'white6', 'white7', 'white8', 'white9']


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Data'
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


    # Extract the data during game
    data = lbs.values_during_game(data)



    # Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
    list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data, old_data)


    # Create a simple graph with all the columns you need to plot
    # lbs.simple_graph(list_with_all_df_separated_by_set, 'pitch', 'yaw', 'roll')

    # Create a scatter with a slider for visualization of target position vs player position
    # lbs.graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set)

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


    # Flatten the list with the spatial errors
    spatial_error, indices = lbs.list_of_five_lists_flatten_list(spatial_error)


    # Create the timestamps without space between sets
    time_stamps_without_between_set_space = lbs.creation_pf_timestamps_without_space_between_sets(list_time_stamp_of_min_spatial_error_separated_by_set)

    # Calculate the simple regression line
    simple_slope,  simple_intercept,  simple_rmse = lbs.simple_linear_regression(time_stamps_without_between_set_space, spatial_error, plot=False)

    # Calculate the optimal number of breakpoints for Segmented regression
    # optimal_aic_n, optimal_bic_n = lbs.determine_the_number_of_breakpoints(time_stamps_without_between_set_space, spatial_error, max_number_of_breakpoints_to_check=15, index_duration=15)

    # Calculate the segmented regression line
    segmented_slopes, segmented_intercepts, segmented_rmse = lbs.segmented_linear_regression(time_stamps_without_between_set_space, spatial_error,
                                                               number_of_breakpoints=1, index_duration=15, plot=False)
    # print(f'segemnted_slopes={segmented_slopes}')
    # print(f'segemnted_intercepts={segmented_intercepts}')
    # print(f'segmented_rmse={segmented_rmse}')
    # print(f'simple_slope={simple_slope}')
    # print(f'simple_intercept={simple_intercept}')
    # print(f'simple_rmse={simple_rmse}')


    # Append values to create the df after with the results
    list_simple_rmse.append(simple_rmse)
    list_simple_slope.append(simple_slope)
    list_simple_intercept.append(simple_intercept)

    list_segmented_rmse.append(segmented_rmse)
    list_segmented_slope_before.append(segmented_slopes[0])
    list_segmented_intercept_before.append(segmented_intercepts[0])
    list_segmented_slope_after.append(segmented_slopes[1])
    list_segmented_intercept_after.append(segmented_intercepts[1])

    average_spatial_error = np.mean(spatial_error)
    list_average_spatial_error.append(average_spatial_error)
    sd_spatial_error = np.std(spatial_error)
    list_sd_spatial_error.append(sd_spatial_error)

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



# Calculate slopes, error, Average Spatial error, and sd spatial error at 500 for everyone
dist = {'ID': list_ID,
        'Simple Regression Slope': list_simple_slope,
        'Simple Regression Intercept': list_simple_intercept,
        'Simple Regression RMSE': list_simple_rmse,
        'Segmented Regression Slope before': list_segmented_slope_before,
        'Segmented Regression Intercept before': list_segmented_intercept_before,
        'Segmented Regression Slope after': list_segmented_slope_after,
        'Segmented Regression Intercept after': list_segmented_intercept_after,
        'Segmented Regression RMSE': list_segmented_rmse,
        'Average Spatial Error all sets': list_average_spatial_error,
        'Sd Spatial Error all sets': list_sd_spatial_error,
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
df = pd.DataFrame(dist)
directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Results'
os.chdir(directory)
df.to_excel('Results.xlsx')


df_long = df.melt(id_vars=['ID'],
                  value_vars=['Average Spatial error set 1', 'Average Spatial error set 2',
                              'Average Spatial error set 3', 'Average Spatial error set 4',
                              'Average Spatial error set 5'],
                  var_name='Set', value_name='Average Spatial Error')  # Ensure correct name

# Rename set names for readability
df_long['Set'] = df_long['Set'].str.replace('Average Spatial error set ', 'Set ')

# Create boxplot
custom_palette = {
    "pink": "#FFC0CB",      # Soft pink
    "static": "#4F4F4F",    # Dark gray
    "white": "#D3D3D3"      # Light gray
}
plt.figure(figsize=(12, 6))
sns.boxplot(x='Set', y='Average Spatial Error', hue='ID', data=df_long, palette=custom_palette)  # Use correct column name

# Customize plot
plt.title('Comparison of Spatial Error Across Sets and ID Groups')
plt.xlabel('Set')
plt.ylabel('Average Spatial Error')
plt.legend(title='ID')

# Show plot
plt.show()

# Plot Slopes
df_melted_Slope = df.melt(id_vars=['ID'], value_vars=[
    'Simple Regression Slope',
    'Segmented Regression Slope before',
    'Segmented Regression Slope after'
], var_name='Regression Type', value_name='Value')

# Set up the plotting style (optional)
sns.set(style="whitegrid")

# Create the boxplot with 3 major columns (regression types) and hues (ID)
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regression Type', y='Value', hue='ID', data=df_melted_Slope, showfliers=False)
plt.title('Slope by ID')
plt.xticks(rotation=35, ha='right')  # Rotate the x-axis labels for better readability
plt.legend(title='ID')
plt.gca().set_xlabel('')  # Remove the "Regression Type" label without leaving space
plt.tight_layout()
plt.show()



# Plot intercepts
df_melted_intercept = df.melt(id_vars=['ID'], value_vars=[
    'Simple Regression Intercept',
    'Segmented Regression Intercept before',
    'Segmented Regression Intercept after'
], var_name='Regression Type', value_name='Value')

# Set up the plotting style (optional)
sns.set(style="whitegrid")

# Create the boxplot with 3 major columns (regression types) and hues (ID)
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regression Type', y='Value', hue='ID', data=df_melted_intercept, showfliers=False)
plt.title('Intercept by ID')
plt.xticks(rotation=35, ha='right')  # Rotate the x-axis labels for better readability
plt.legend(title='ID')
plt.gca().set_xlabel('')  # Remove the "Regression Type" label without leaving space
plt.tight_layout()
plt.show()



# Plot RMSE
df_melted_rmse = df.melt(id_vars=['ID'], value_vars=[
    'Simple Regression RMSE',
    'Segmented Regression RMSE'
], var_name='Regression Type', value_name='Value')

# Set up the plotting style (optional)
sns.set(style="whitegrid")

# Create the boxplot with 3 major columns (regression types) and hues (ID)
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regression Type', y='Value', hue='ID', data=df_melted_rmse, showfliers=False)
plt.title('RMSE by ID')
plt.xticks(rotation=35, ha='right')  # Rotate the x-axis labels for better readability
plt.legend(title='ID')
plt.gca().set_xlabel('')  # Remove the "Regression Type" label without leaving space
plt.tight_layout()
plt.show()


# Plot Average Spatial Error
plt.figure(figsize=(10, 6))
colors = ['lightblue', 'lightgreen', 'lightcoral']  # Adjust colors as needed
sns.boxplot(data=df, x='ID', y='Average Spatial Error all sets', hue='ID', palette=colors, legend=False)
plt.title('Average Spatial Error by ID')
plt.xlabel('ID')
plt.ylabel('Average Spatial Error')
plt.show()



# Plot Sd Spatial Error
plt.figure(figsize=(10, 6))
colors = ['lightblue', 'lightgreen', 'lightcoral']  # Adjust colors as needed
sns.boxplot(data=df, x='ID', y='Sd Spatial Error all sets', palette=colors)
plt.title('Sd Spatial Error by ID')
plt.xlabel('ID')
plt.ylabel('Sd Spatial Error')
plt.show()



