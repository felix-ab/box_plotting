#hrv modularized
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
import csv
from scikit_posthocs import posthoc_tukey
from statannotations.Annotator import Annotator
import math

#Reading hrv data
def read_hrv_data(hrv_folder_path, hrv_2022_path):
    hrv_files = [f for f in os.listdir(hrv_folder_path) if f.endswith('.csv')]
    new_df = pd.DataFrame(columns=['date','patient id','quality','HR','AVNN', 'SDNN', 'RMSSD','PNN50'])
    for file in hrv_files:
        file_path = os.path.join(hrv_folder_path, file)
        with open(file_path, 'r') as file:
            df = pd.read_csv(file, skiprows=1, usecols= ['date','patient id','quality','HR','AVNN', 'SDNN', 'RMSSD','PNN50'])
            #Find the highest quality measurement
            best_row= df['quality'].idxmax()
            # Extract the desired row
        new_row = df.loc[best_row].to_frame().T
        #concatinate the desired row to the new dataframe
        new_df = pd.concat([new_df, new_row], ignore_index=True)
    with open(hrv_2022_path, 'r') as file:
        raw_df = pd.read_csv(file, usecols= ['date','patient id','quality','HR','AVNN', 'SDNN', 'RMSSD','PNN50'])
    # group the dataframe by age and find the index of the maximum score for each age
    max_quality_idx = raw_df.groupby('patient id')['quality'].idxmax()
    # use the index to extract the corresponding height value for each maximum score
    df_best = raw_df.loc[max_quality_idx, ['date','patient id','quality','HR','AVNN', 'SDNN', 'RMSSD','PNN50']]
    new_df = pd.concat([new_df, df_best], ignore_index=True)
    new_df['patient id'] = new_df['patient id'].astype(int)
    new_df = new_df.rename(columns={'patient id': 'patient_id'})
    return new_df

#Reading demographics data
def read_demographics_data(demographics_folder_path,demographics_2022,gender_demographics_2021):
    sport_dict = {}
    birthday_dict = {}
    name_dict = {}
    gender_dict = {}
    grade_dict = {}
    #reading 2021 data
    demographics_files = [f for f in os.listdir(demographics_folder_path) if f.endswith('.csv')]
    for filename in demographics_files:
        file_path = os.path.join(demographics_folder_path, filename)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            row1 = next(reader)
            Patient = int(row1[0])
            if row1[8] == '[not completed]':
                birthday = '2021-01-01'
            else:
                birthday = row1[8]
            name = str(row1[6]) + ' ' + str(row1[7])
            if row1[9][:2].isnumeric() == True:
                grade = 2000 + int(row1[9][:2])
            elif row1[11] != '':
                grade = row1[11]
            else:
                grade = 'unknown'
            Sport = row1[13]
            sport_dict[Patient] = Sport
            birthday_dict[Patient] = birthday
            name_dict[Patient] = name
            grade_dict[Patient] = grade
    #adding gender for 2021 data
    with open(gender_demographics_2021, 'r') as file1:
        reader = csv.reader(file1)
        next(reader)
        for row in reader:
            Patient = int(row[0])
            gender = row[1]
            gender_dict[Patient] = gender
    #reading 2022 data
    with open(demographics_2022, 'r') as file2:
        reader = csv.reader(file2)
        next(reader)
        for row in reader:
            Patient = 1000 + int(row[0])
            if row[6] == '':
                name = 'No_Name'
            else:
                name = str(row[6]) + ' ' + str(row[7])
            if row[10] == '[not completed]' or row[10] == '':
                birthday = '2021-01-01'
            else:
                birthday = row[10]
            #birthday = row[10]
            Sport = row[16]
            gender = row[9]
            if row[11][:2].isnumeric() == True:
                grade = 2000 + int(row[11][:2])
            elif row[13] != '':
                grade = row[13]
            else:
                grade = 'unknown'
            sport_dict[Patient] = Sport
            birthday_dict[Patient] = birthday
            name_dict[Patient] = name
            gender_dict[Patient] = gender 
            grade_dict[Patient] = grade 
    dict_list = [birthday_dict, sport_dict, name_dict, gender_dict, grade_dict]
    return dict_list  

def add_demographics_data(new_df, dict_list):
    new_df['patient_id'] = new_df['patient_id'].astype(int)
    for i, row in new_df.iterrows():
        id = row['patient_id']
        id = int(id)
        if id in dict_list[0]:
            birthday_temp = dict_list[0].get(id)
            new_df.at[i,'birthday'] = birthday_temp
        else:
            new_df.at[i,'birthday'] = '2021-01-01'
        if id in dict_list[1]:
            sport_temp = dict_list[1].get(id)
            new_df.at[i,'Sport'] = sport_temp
        else:
            new_df.at[i,'Sport'] = 'No_Sport'
        if id in dict_list[2]:
            name_temp = dict_list[2].get(id)
            new_df.at[i,'name'] = name_temp
        else:
            new_df.at[i,'name'] = 'No_Name'
        if id in dict_list[3]:
            gender_temp = dict_list[3].get(id)
            new_df.at[i,'gender'] = gender_temp
        else:
            new_df.at[i,'gender'] = 'unknown'
        if id in dict_list[4]:
            grade_temp = dict_list[4].get(id)
            new_df.at[i,'grade'] = grade_temp
        else:
            new_df.at[i,'grade'] = 'unknown'
    # Convert the 'date' column to a datetime object
    new_df['birthday'] = pd.to_datetime(new_df['birthday'], infer_datetime_format=True)
    new_df['date'] = pd.to_datetime(new_df['date'], infer_datetime_format=True)
    # Extract the year from the date column
    new_df['birth_year'] = new_df['birthday'].dt.year 
    # puts all mistake birth years into the same category
    new_df['birth_year'] = new_df['birth_year'].apply(lambda x: 2023 if x > 2020 else x)
    # Convert the DataFrame to an Excel file and export it
    new_df.to_excel('all_hrv_2.xlsx', index=False)

    #group the sport column by its unique values and count the occurrences
    counts = new_df.groupby('Sport').size()
    # identify 'Sports' values that appear less than 3 times
    mask = new_df['Sport'].isin(counts[counts < 3].index)
    # replace those values with 'other'
    new_df.loc[mask, 'Sport'] = 'Other'
    #replace "No_Sport" with "other"
    new_df.loc[new_df['Sport'] == 'No_Sport', 'Sport'] = 'Other'
    return new_df

def group_data_by_variable(new_df, group_var, hr_var):
    grouped_data = []
    unique_groups = np.unique(new_df[group_var])
    for group in unique_groups:
        grouped_data.append(new_df[new_df[group_var] == group][hr_var])
    if group_var == 'Sport':
        idx = np.where(unique_groups == 'Other')[0][0]
        unique_groups = np.concatenate((np.delete(unique_groups, idx), ['Other']))
    return unique_groups, grouped_data

def stats_and_plotting(unique_groups,grouped_data,new_df,hr_var,group_var, counter):
    save_path = '/Users/felixbrener/Documents/saved_figures/'
    f_result= stats.f_oneway(*grouped_data)
    plt.figure(counter, figsize=(13, 8))
    #removing erronoius measurements
    if hr_var == 'PNN50':
        new_df = new_df[new_df[hr_var] >= 0]
        minimumval = ' values less than\nzero '
    else:
        new_df = new_df[new_df[hr_var] > 0]
        minimumval = ' values of zero or\nless '
    #remove duplicates so that the newest one shows up
    new_df.sort_values(by=['date'])
    new_df = new_df.copy()
    new_df.loc[:, 'name'] = new_df['name'].str.lower()
    new_df = new_df.drop_duplicates(subset='name', keep='last')

    if  f_result.pvalue < 0.05:
        #Performing Tukey's HSD test
        tukey_df = posthoc_tukey(new_df, val_col=hr_var, group_col=group_var)
        #checking result of posthoc_tukey's
        #comverting the matrix to a non-redundant list of comparisons with the p-value
        remove = np.tril(np.ones(tukey_df.shape), k=0).astype("bool")
        tukey_df[remove] = np.nan
        #removing the lower half and diagonal of the matrix and turning the matrix format into a long dataframe
        molten_df = tukey_df.melt(ignore_index=False).reset_index().dropna()
        #filtering the rows where the value is less than 0.05
        filtered_rows = molten_df[molten_df['value'] < 0.05]
        #creating box plots
        ax = sns.boxplot(data=new_df, x=group_var, y=hr_var, order=unique_groups)
        if hr_var == 'RMSSD' or hr_var =='SDNN':
            ax.set(xlabel =group_var, ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel =group_var, ylabel = hr_var, title = hr_var +' by '+group_var)
        
        ax.text(0, 1.006, "p-value annotation legend:\nNot Significant: p > 5.00e-02\n*: 1.00e-02 < p <= 5.00e-02\n**: 1.00e-03 < p <= 1.00e-02\n***: 1.00e-04 < p <= 1.00e-03\n****: p <= 1.00e-04", transform=ax.transAxes, fontsize=9)
        ax.text(0.8, 1.006, "n = " + str(len(new_df)) + " subjects\nAll subjects with "+str(hr_var)+ minimumval + "have been removed from the dataset", transform=ax.transAxes, fontsize=9)
        if len(filtered_rows) > 0:
            pairs = [(i[1]["index"], i[1]["variable"]) for i in filtered_rows.iterrows()]
            p_values = [i[1]["value"] for i in filtered_rows.iterrows()]

            annotator = Annotator(ax, pairs, data=new_df, x=group_var, y=hr_var, order=unique_groups)
            annotator.configure(text_format="star", loc="inside")
            annotator.set_pvalues_and_annotate(p_values)
        
        plt.title(hr_var +' by '+ group_var)
        if len(unique_groups) > 10:
            plt.xticks(rotation=15)
        plt.axhline(y=np.mean(new_df[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        plt.legend()
        if counter == 8:
            plt.show()
        else:
            plt.show(block = False)
        
    else:
        print('Unable to reject hypothesis')
        
        ax = sns.boxplot(data=new_df, x=group_var, y=hr_var, order=unique_groups)
        if hr_var == 'RMSSD' or hr_var =='SDNN':
            ax.set(xlabel =group_var, ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel =group_var, ylabel = hr_var, title = hr_var +' by '+group_var)
        plt.title(hr_var +' by '+ group_var)
        ax.text(0.89, 1.006, "n = " + str(len(new_df)) + " subjects", transform=ax.transAxes, fontsize=9)
        labels_fixed = list(unique_groups[:-1]) + ['unknown']
        plt.xticks(range(len(unique_groups)), labels_fixed)
        plt.axhline(y=np.mean(new_df[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        plt.legend()
        if counter == 16:
            plt.show(block = True)
        else:
            plt.show(block = False)
    # save the figure
    fig_name = 'figure_' + str(counter) + '.png'  # change the file name as per your preference
    plt.savefig(save_path + fig_name)


if __name__ == '__main__':

    hrv_folder_path = "/Users/felixbrener/Documents/hrv_excel"
    demographics_folder_path ='/Users/felixbrener/Documents/demographics_excel'
    hrv_2022_path = '/Users/felixbrener/Documents/ARC/hrv_2022.csv'
    demographics_2022 = '/Users/felixbrener/Downloads/PlaySafety20222023_DATA_2023-02-27_1040.csv'
    gender_demographics_2021 = '/Users/felixbrener/Documents/ARC/gender_2021_demographics.csv'

    hrv_list = ['HR','AVNN', 'SDNN', 'RMSSD','PNN50']
    group_list = ['Sport','birth_year','gender']
    counter = 0

    #reads in the hrv and demographic data 
    new_df  = read_hrv_data(hrv_folder_path, hrv_2022_path)
    dict_list = read_demographics_data(demographics_folder_path, demographics_2022, gender_demographics_2021)
    #add demographics data to new_df
    #adds sport and birthday data to new_df
    new_df = add_demographics_data(new_df, dict_list)
    for group in group_list:
        group_var = group
        for hrv in hrv_list:
            hr_var = hrv

            #finds all unique values in demographics data columns
            unique_groups, grouped_data = group_data_by_variable(new_df, group_var, hr_var)

            #performs one way ANOVa and tukey's HSD and then created box plots showing significant differences
            stats_and_plotting(unique_groups,grouped_data,new_df,hr_var,group_var, counter)
            counter = counter + 1

            #stats_and_plotting calls group_data_by_variable
            #group_data_by_variable calls add_demographics_data
            #add_demographics_data calls read_demographics_data and read_hrv_data 
            #group_data_by_variable()
