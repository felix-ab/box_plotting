#analysis of top athletes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math

def get_top_names(df, hrv_list):
    # Drop rows with empty cells in the name column
    df = df[df['name'] != 'No_Name']
    df['name'] = df['name'].str.lower().str.replace(' ', '')
    df['birthday'] = pd.to_datetime(df['birthday'])
    df['date'] = pd.to_datetime(df['date'])
    # Calculate the age in years
    age = (df['date'] - df['birthday']).dt.total_seconds() / (365.25 * 24 * 3600)
    df['age'] = age
    df['age'] = np.floor(df['age']).astype(int)
    df.loc[df['age'] < 5, 'age'] = 'unknown'
    df['age'] = df['age'].astype(str)
    df = df[df['HR'] != 0]

    # Select the desired columns
    columns_of_interest = ['name', 'HR', 'SDNN', 'RMSSD']#'AVNN', 'PNN50'
    df_selected = df[columns_of_interest]
    df_selected['HR'] = np.log(df_selected['HR'].astype(float))
    df_selected['SDNN'] = np.log(df_selected['SDNN'].astype(float))
    df_selected['RMSSD'] = np.log(df_selected['RMSSD'].astype(float))

    # Normalize the heart rate variable columns
    df_normalized = (df_selected.iloc[:, 1:] - df_selected.iloc[:, 1:].min()) / (df_selected.iloc[:, 1:].max() - df_selected.iloc[:, 1:].min())

    # Calculate the rank of each name within each heart rate variable
    rank_df = df_normalized.rank(ascending=False)

    # Convert the rank to a percentage score
    percentage_df = (rank_df / df_normalized.shape[0]) * 100
    #inverting the HR percentages because lower HR is better
    for hrv in hrv_list[1:]:
        percentage_df[hrv] = (percentage_df[hrv] - 100).abs()

    # Calculate the average across the numerical data columns
    percentage_df['hrv_percentile'] = percentage_df.iloc[:, 1:].mean(axis=1)


    percentage_df =  percentage_df.drop(percentage_df.columns[1:6], axis=1)

    percentage_df['name'] = df['name']

    # Calculate the threshold value for the top 5th percentile
    threshold = percentage_df['HR'].quantile(0.95)

    # Filter the DataFrame to include only names in the top 5th percentile
    top_names = percentage_df[percentage_df['HR'] >= threshold]['name'].tolist()
    print(top_names)
    return top_names

def clean_up_df(df):
    # Drop rows with empty cells in the name column
    df = df[df['name'] != 'No_Name']
    df['name'] = df['name'].str.lower().str.replace(' ', '')
    df['birthday'] = pd.to_datetime(df['birthday'])
    df['date'] = pd.to_datetime(df['date'])
    # Calculate the age in years
    age = (df['date'] - df['birthday']).dt.total_seconds() / (365.25 * 24 * 3600)
    df['age'] = age
    df['age'] = np.floor(df['age']).astype(int)
    df.loc[df['age'] < 5, 'age'] = 100
    df = df[(df['HR'] > 55) & (df['HR'] <= 100)]

    #Group by 'name' and keep the row with the largest 'date'
    df = df.loc[df.groupby('name')['date'].idxmax()]
    
    # Count the frequency of 'Sport' values
    sport_counts = df['Sport'].value_counts()
    # Identify values that appear less than 4 times
    values_to_replace = sport_counts[sport_counts < 4].index.tolist()
    # Replace values with 'other'
    df.loc[df['Sport'].isin(values_to_replace), 'Sport'] = 'Other'
    df.loc[df['Sport']=='No_Sport'] = 'Other'
    
    
    # Replace 'boy's basketball' with 'boys basketball'
    df['Sport'] = df['Sport'].replace("Boy's Basketball", "Boys Basketball")
    # Replace 'girl's basketball' with 'girls basketball'
    df['Sport'] = df['Sport'].replace("Girl's Basketball", "Girls Basketball")

    #sport_order = {'Boys Basketball': 0, 'Girls Basketball': 1,'Boys Soccer': 2,
    #'Girls Soccer': 3,'Cross Country': 4, 'Field Hockey': 5,'Football': 6,'Ice Hockey': 7,'Squash': 8,
    #'Tennis': 9,'Volleyball': 10,'Water Polo': 11,'Other': 12}


    return df

def stats_plotting(filtered_df, title, hrv_list, top_names):
    filtered_df['hue'] = filtered_df['name'].apply(lambda x: 'top athlete' if x in top_names else 'not top athlete')
    hrv_list= ['HR', 'SDNN', 'RMSSD']#'AVNN' , 'PNN50'

    fig, axs = plt.subplots(1, len(hrv_list), figsize=(20, 5))

    # Create swarm plots for each set of axes
    for i, column_name in enumerate(hrv_list):
        if len(filtered_df) > 0:
            print(filtered_df[column_name])
            #filtered_df[column_name] = filtered_df[column_name].astype(float)
            #column = filtered_df[column_name]
            #sns.swarmplot(data=filtered_df, y=column_name, hue= 'hue',ax=axs[i])
            # Define the value-color mapping
            value_color_mapping = {'top athlete': '#ff7f0e', 'not top athlete': '#1f77b4'}
            filtered_df = filtered_df.astype({column_name: float})
            sns.violinplot(y=filtered_df[column_name],color = '#E1E1E1', linewidth= 0, ax=axs[i])#x=[""]*len(filtered_df), data= filtered_df[column_name]
            sns.swarmplot(x=[""]*len(filtered_df),y=filtered_df[column_name], hue= filtered_df['hue'],palette=value_color_mapping,ax=axs[i])

            axs[i].set_title(f'Swarm Plot ({column_name})')
            axs[i].set_xlabel(column_name)
            #axs[i].set_ylabel('Value')
            if column_name == 'HR':
                    axs[i].set_ylabel(f'{column_name} (bpm)')
            else:
                axs[i].set_ylabel(f'{column_name} (ms)' if column_name != 'PNN50' else f'{column_name} (%)')

            # Add a title to the entire figure
            fig.suptitle(title)
        else:
            print('too short')

    # Adjust the layout and display the plot
    plt.subplots_adjust(wspace=0.4)
    additional_text = "all_athletes_"
    filename = f'{additional_text}{title.replace(" ", "_")}.png'
    
    plt.savefig(save_path + filename)

    # Close the figure
    plt.close(fig)
    #plt.show()



if __name__ == '__main__':

    df1 = pd.read_excel('/Users/felixbrener/Documents/ARC/all_hrv.xlsx')
    #top_athletes_df= pd.read_excel('/Users/felixbrener/top_athletes2.xlsx')
    hrv_list = ['HR', 'SDNN', 'RMSSD']#'AVNN','PNN50'
    gender_list = ['male','female','unknown']
    save_path = '/Users/felixbrener/Documents/hr_sdnn_rmssd/'
    
'''   
    df1 = clean_up_df(df1)
    sports = df1['Sport'].unique()
    for sport in sports:
        df= df1[df1['Sport'] == sport].reset_index(drop=True)
        #returns the top 5% best athletes in that sport
        top_names = get_top_names(df, hrv_list)

        #df, sports_with_genders, unique_sports = make_titles(df)
        for gender in gender_list:
            title = str(sport+' '+gender)
            # Filter rows based on conditions
            filtered_df = df[(df['Sport'] == sport) & (df['gender'] == gender)]
            filtered_df = filtered_df[filtered_df['HR'] > 0].reset_index(drop=True)
            #filtered_df = filtered_df.reset_index(drop=True)
            if len(filtered_df) > 3:
                stats_plotting(filtered_df, title, hrv_list, top_names)

    

df = clean_up_df(df1)
top_names = get_top_names(df, hrv_list)
for gender in gender_list:
    title = str(gender)
    # Filter rows based on conditions
    filtered_df = df[df['gender'] == gender]
    filtered_df = filtered_df[filtered_df['HR'] > 0].reset_index(drop=True)
    #filtered_df = filtered_df.reset_index(drop=True)
    if len(filtered_df) > 3:
        stats_plotting(filtered_df, title, hrv_list, top_names)

'''
df = clean_up_df(df1)
top_names = get_top_names(df, hrv_list)
title = 'All Athletes less than 100 hr and log taken'
# Filter rows based on conditions
filtered_df = df[df['HR'] > 0].reset_index(drop=True)
#filtered_df = filtered_df.reset_index(drop=True)
stats_plotting(filtered_df, title, hrv_list, top_names)
