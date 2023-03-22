#same_patients_over_time
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

def find_duplicates(all_hrv):
    all_hrv_df = pd.read_excel(all_hrv)
    all_hrv_df.loc[:, 'name'] = all_hrv_df['name'].str.lower()
    #removing spaces from the 'name' column so i can sort more easily
    all_hrv_df['name'] = all_hrv_df['name'].str.replace('\s+', '')
    #creating a column that lists recording date by year
    all_hrv_df['date_year'] = all_hrv_df['date'].dt.year
    
    return all_hrv_df


def group_data_by_variable(all_hrv_df, group_var, hr_var):
    grouped_data = []
    unique_groups = np.unique(all_hrv_df[group_var])
    for group in unique_groups:
        grouped_data.append(all_hrv_df[all_hrv_df[group_var] == group][hr_var])
    return unique_groups, grouped_data

def stats_and_plotting(unique_groups, grouped_data, all_hrv_df, hr_var, group_var, counter):
    save_path = '/Users/felixbrener/Documents/saved_figures/'
    f_result= stats.f_oneway(*grouped_data)
    plt.figure(counter, figsize=(13, 8))
    #removing erronoius measurements
    if hr_var == 'PNN50':
        all_hrv_df = all_hrv_df[all_hrv_df[hr_var] >= 0]
        minimumval = ' values less than\nzero '
    else:
        all_hrv_df = all_hrv_df[all_hrv_df[hr_var] > 0]
        minimumval = ' values of zero or\nless '
    #remove rows with names that only appear once
    all_hrv_df = all_hrv_df.copy()
    all_hrv_df.drop_duplicates(subset=['name', 'date_year'], keep='last', inplace=True)
    duplicates = all_hrv_df.copy()
    # group the name column by its unique values and count the occurrences
    counts = duplicates.groupby('name').size()
    # filter for values that appear only once
    values_to_delete = counts[counts == 1].index.tolist()
    # filter the DataFrame to keep only rows that do not contain those values
    duplicates = duplicates[~duplicates['name'].isin(values_to_delete)]
    duplicates.sort_values(by=['name'], ascending=False, inplace=True)
    #duplicates.drop_duplicates(subset=['date_year', 'name'], keep='last', inplace=True)
    #convertign data column from stirngs to floats
    duplicates = duplicates.astype({'quality': float, 'HR': float, 'AVNN': float, 'SDNN': float, 'RMSSD': float, 'PNN50': float})

    if  f_result.pvalue < 0.05:
        #Performing Tukey's HSD test
        tukey_df = posthoc_tukey(duplicates, val_col=hr_var, group_col=group_var)
        #checking result of posthoc_tukey's
        #comverting the matrix to a non-redundant list of comparisons with the p-value
        remove = np.tril(np.ones(tukey_df.shape), k=0).astype("bool")
        tukey_df[remove] = np.nan
        #removing the lower half and diagonal of the matrix and turning the matrix format into a long dataframe
        molten_df = tukey_df.melt(ignore_index=False).reset_index().dropna()
        #filtering the rows where the value is less than 0.05
        filtered_rows = molten_df[molten_df['value'] < 0.05]
        #creating box plots
        ax = sns.boxplot(data=duplicates, x=group_var, y=hr_var, order=unique_groups)
        if hr_var == 'RMSSD' or hr_var =='SDNN':
            ax.set(xlabel ='year', ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel ='year', ylabel = hr_var, title = hr_var +' by '+ group_var)
        
        ax.text(0, 1.006, "p-value annotation legend:\nNot Significant: p > 5.00e-02\n*: 1.00e-02 < p <= 5.00e-02\n**: 1.00e-03 < p <= 1.00e-02\n***: 1.00e-04 < p <= 1.00e-03\n****: p <= 1.00e-04", transform=ax.transAxes, fontsize=9)
        ax.text(0.8, 1.006, "n = " + str(len(duplicates)) + " subjects\nAll subjects with "+str(hr_var)+ minimumval + "have been removed from the dataset", transform=ax.transAxes, fontsize=9)
        if len(filtered_rows) > 0:
            pairs = [(i[1]["index"], i[1]["variable"]) for i in filtered_rows.iterrows()]
            p_values = [i[1]["value"] for i in filtered_rows.iterrows()]

            annotator = Annotator(ax, pairs, data=duplicates, x=group_var, y=hr_var, order=unique_groups)
            annotator.configure(text_format="star", loc="inside")
            annotator.set_pvalues_and_annotate(p_values)
        
        plt.title(hr_var +' by date of recording')
        if len(unique_groups) > 10:
            plt.xticks(rotation=15)
        plt.axhline(y=np.mean(duplicates[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        plt.legend()
        if counter == 5:
            plt.show()
        else:
            plt.show(block = False)
        
    else:
        print('Unable to reject hypothesis')
        
        ax = sns.boxplot(data=duplicates, x=group_var, y=hr_var, order=unique_groups)
        if hr_var == 'RMSSD' or hr_var =='SDNN':
            ax.set(xlabel ='year', ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel ='year', ylabel = hr_var, title = hr_var +' by '+group_var)
        plt.title(hr_var +' by date of recording')
        ax.text(0.89, 1.006, "n = " + str(len(duplicates)) + " subjects", transform=ax.transAxes, fontsize=9)
        #labels_fixed = list(unique_groups[:-1]) + ['unknown']
        #plt.xticks(range(len(unique_groups)), labels_fixed)
        plt.axhline(y=np.mean(duplicates[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        plt.legend()
        if counter == 5:
            plt.show(block = True)
        else:
            plt.show(block = False)
    # save the figure
    fig_name = 'plotting_' + str(counter) + '.png'  # change the file name as per your preference
    plt.savefig(save_path + fig_name)


if __name__ == '__main__':

    all_hrv = '/Users/felixbrener/Documents/ARC/all_hrv.xlsx'
    hrv_list = ['HR','AVNN', 'SDNN', 'RMSSD','PNN50']
    group_var = 'date_year'
    counter = 0

 
    for hrv in hrv_list:
        hr_var = hrv

        #finds all duplicate subjects in the data
        all_hrv_df = find_duplicates(all_hrv)
        #finds all unique values in demographics data columns
        unique_groups, grouped_data = group_data_by_variable(all_hrv_df, group_var, hr_var)
        #performs one way ANOVa and tukey's HSD and then created box plots showing significant differences
        stats_and_plotting(unique_groups,grouped_data, all_hrv_df, hr_var,group_var, counter)
        counter = counter + 1

        #stats_and_plotting calls group_data_by_variable
        #group_data_by_variable calls add_demographics_data
        #add_demographics_data calls read_demographics_data and read_hrv_data 
        #group_data_by_variable()
