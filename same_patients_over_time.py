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
    all_hrv_df['Sport'] = all_hrv_df['Sport'].str.replace("Girl's", "Girls")
    all_hrv_df['Sport'] = all_hrv_df['Sport'].str.replace("Boy's", "Boys")
    return all_hrv_df


def group_data_by_variable(all_hrv_df, group_var, hr_var):
    grouped_data = []
    unique_groups = np.unique(all_hrv_df[group_var])
    for group in unique_groups:
        grouped_data.append(all_hrv_df[all_hrv_df[group_var] == group][hr_var])
    if group_var == 'Sport':
        sport_order = {'Boys Basketball': 0, 'Girls Basketball': 1,'Boys Soccer': 2,
        'Girls Soccer': 3,'Cross Country': 4, 'Field Hockey': 5,'Football': 6,'Ice Hockey': 7,'Squash': 8,
        'Tennis': 9,'Volleyball': 10,'Water Polo': 11,'Basketball': 12, 'No_Sport': 13, 'Other': 14}
        unique_groups = sorted(unique_groups, key=lambda s: sport_order[s])
    return unique_groups, grouped_data

def stats_and_plotting(unique_groups, grouped_data, all_hrv_df, hr_var, group_var, counter):
    save_path = '/Users/felixbrener/Documents/violin_swarm2/'
    all_hrv_df[['quality', 'HR', 'AVNN', 'SDNN','RMSSD','PNN50']] = all_hrv_df[['quality', 'HR', 'AVNN', 'SDNN','RMSSD','PNN50']].apply(pd.to_numeric)
    #f_result = stats.f_oneway(*grouped_data)
    plt.figure(counter, figsize=(13, 8))
    #removing erronoius measurements
    if hr_var == 'PNN50':
        all_hrv_df = all_hrv_df[all_hrv_df[hr_var] >= 0]
        minimumval = ' values less than\nzero '
    else:
        all_hrv_df = all_hrv_df[all_hrv_df[hr_var] > 0]
        minimumval = ' values of zero or\nless '
    #all_hrv_df = all_hrv_df.copy()
    #remove rows with names that only appear in one year
    all_hrv_df = all_hrv_df[all_hrv_df['name'].isin(all_hrv_df.loc[all_hrv_df['date_year']==2021, 'name']) & all_hrv_df['name'].isin(all_hrv_df.loc[all_hrv_df['date_year']==2022, 'name'])]
    #remove duplicates with same name, same yeartimes
    duplicates = all_hrv_df.drop_duplicates(subset=['name', 'date_year'], keep='last')
    #converting data column from strings to floats
    duplicates = duplicates.astype({'quality': float, 'HR': float, 'AVNN': float, 'SDNN': float, 'RMSSD': float, 'PNN50': float})
    #diff_df = duplicates.groupby('name')[hr_var].diff().dropna().reset_index()
    diff_df = duplicates.groupby('name').apply(lambda x: x[[hr_var]].diff().dropna().assign(name=x['name'])).reset_index(drop=True)
    last_vals = duplicates.groupby('name').last().drop(columns=hr_var)
    diff_df = diff_df.join(last_vals, on='name')
    #group the sport column by its unique values and count the occurrences
    counts = diff_df.groupby('Sport').size()
    # identify 'Sports' values that appear less than 3 times
    mask = diff_df['Sport'].isin(counts[counts < 3].index)
    # replace those values with 'other'
    diff_df.loc[mask, 'Sport'] = 'Other'
    diff_df.to_excel('differences.xlsx', index=False)
    diff_grouped_data = []
    diff_unique_groups = np.unique(diff_df[group_var])
    
    for group in diff_unique_groups:
        diff_grouped_data.append(diff_df[diff_df[group_var] == group][hr_var])
    f_result_diff = stats.f_oneway(*diff_grouped_data)
    if group_var == 'Sport':
        sport_order = {'Boys Basketball': 0, 'Girls Basketball': 1,'Boys Soccer': 2,
        'Girls Soccer': 3,'Cross Country': 4, 'Field Hockey': 5,'Football': 6,'Ice Hockey': 7,'Squash': 8,
        'Tennis': 9,'Volleyball': 10,'Water Polo': 11,'Basketball': 12, 'Other': 13}
        diff_unique_groups = sorted(diff_unique_groups, key=lambda s: sport_order[s])
    
    if  f_result_diff.pvalue < 0.05:
        #Performing Tukey's HSD test
        tukey_df = posthoc_tukey(diff_df, val_col=hr_var, group_col=group_var)
        #checking result of posthoc_tukey's
        #comverting the matrix to a non-redundant list of comparisons with the p-value
        remove = np.tril(np.ones(tukey_df.shape), k=0).astype("bool")
        tukey_df[remove] = np.nan
        #removing the lower half and diagonal of the matrix and turning the matrix format into a long dataframe
        molten_df = tukey_df.melt(ignore_index=False).reset_index().dropna()
        #filtering the rows where the value is less than 0.05
        filtered_rows = molten_df[molten_df['value'] < 0.05]
        ax = sns.violinplot(data=diff_df, x=group_var, y=hr_var, order=unique_groups, color = '#E1E1E1', linewidth= 0)
        ax = sns.swarmplot(data=diff_df, x=group_var, y=hr_var, order=unique_groups, alpha = 0.7)
        #ax = sns.swarmplot(data=diff_df, x=group_var, y=hr_var, order=diff_unique_groups, alpha = 0.65)
        # add pointplot with average values
        ax = sns.pointplot(data=diff_df, x=group_var, y=hr_var, color="black", markers="d", scale=1.5, join=False, ci=None, size=10, order=diff_unique_groups, linewidth=2, label='Group Average')
        if hr_var == 'RMSSD' or hr_var =='SDNN' or hr_var =='AVNN':
            ax.set(xlabel =group_var, ylabel = 'Difference in' + hr_var + "(ms)", title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        elif hr_var == 'HR':
            ax.set(xlabel =group_var, ylabel = 'Difference in' + hr_var + "(bpm)", title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        elif hr_var == 'PNN50':
            ax.set(xlabel =group_var, ylabel = 'Difference in' + hr_var + "(percentage)", title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        else:
            ax.set(xlabel =group_var, ylabel = 'Difference in ' + hr_var, title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        if group_var == 'gender':
            ax.set(xlabel='Gender')
        ax.text(0, 1.006, "p-value annotation legend:\nNot Significant: p > 5.00e-02\n*: 1.00e-02 < p <= 5.00e-02\n**: 1.00e-03 < p <= 1.00e-02\n***: 1.00e-04 < p <= 1.00e-03\n****: p <= 1.00e-04", transform=ax.transAxes, fontsize=9)
        ax.text(0.8, 1.006, "n = " + str(len(diff_df)) + " subjects\nAll subjects with "+str(hr_var)+ minimumval + "have been removed from the dataset", transform=ax.transAxes, fontsize=9)
        if len(filtered_rows) > 0:
            pairs = [(i[1]["index"], i[1]["variable"]) for i in filtered_rows.iterrows()]
            p_values = [i[1]["value"] for i in filtered_rows.iterrows()]

            annotator = Annotator(ax, pairs, data=diff_df, x=group_var, y=hr_var, order=diff_unique_groups)
            annotator.configure(text_format="star", loc="inside")
            annotator.set_pvalues_and_annotate(p_values)
        
        plt.title(hr_var + ' difference between 2021 and 2022 by ' + group_var)
        if len(diff_unique_groups) > 10:
            plt.xticks(rotation=15)
        if len(ax.get_yticks()) < 10:
            tick_interval = (ax.get_yticks()[-1] - ax.get_yticks()[0]) / (len(ax.get_yticks()) - 1)
            new_ticks = [ax.get_yticks()[0] + i*tick_interval/2 for i in range(2*len(ax.get_yticks())-1)]
            ax.set_yticks(new_ticks)
        plt.axhline(y=np.mean(diff_df[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        # count the number of data points in each group
        counts = diff_df[group_var].value_counts()
        # create a dictionary to map the counts to the group names
        count_dict = dict(zip(counts.index, counts.values))
        # add the counts to the x-tick labels
        if group_var == 'birth_year':
            diff_unique_groups = ['unknown' if x == 2023 else x for x in diff_unique_groups]
            count_dict['unknown'] = count_dict.pop(2023)
        xtick_labels = [f'{name}\n({count_dict[name]})' for name in diff_unique_groups]
        ax.set_xticklabels(xtick_labels)

        plt.legend()
        if counter == 15:
            plt.show()
        else:
            plt.show(block = False)
        
    else:
        print('Unable to reject hypothesis')
        ax = sns.violinplot(data=diff_df, x=group_var, y=hr_var, order=unique_groups, color = '#E1E1E1', linewidth= 0)
        ax = sns.swarmplot(data=diff_df, x=group_var, y=hr_var, order=unique_groups, alpha = 0.7)
        #ax = sns.swarmplot(data=diff_df, x=group_var, y=hr_var, order=diff_unique_groups, alpha = 0.65)
        # add pointplot with average values
        ax = sns.pointplot(data=diff_df, x=group_var, y=hr_var, color="black", markers="d", scale=1.5, join=False, ci=None, size=10, order=diff_unique_groups, linewidth=2, label='Group Average')
        if hr_var == 'RMSSD' or hr_var =='SDNN' or hr_var =='AVNN':
            ax.set(xlabel =group_var, ylabel = 'Difference in ' + hr_var + "(ms)", title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        elif hr_var == 'HR':
            ax.set(xlabel =group_var, ylabel = 'Difference in' + hr_var + "(bpm)", title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        elif hr_var == 'PNN50':
            ax.set(xlabel =group_var, ylabel = 'Difference in' + hr_var + "(percentage)", title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        else:
            ax.set(xlabel =group_var, ylabel = 'Difference in ' + hr_var, title = hr_var + ' difference between 2021 and 2022 by ' + group_var)
        if group_var == 'gender':
            ax.set(xlabel='Gender')
        plt.title(hr_var + ' difference between 2021 and 2022 by ' + group_var)
        ax.text(0.89, 1.006, "n = " + str(len(diff_df)) + " subjects", transform=ax.transAxes, fontsize=9)
        #labels_fixed = list(unique_groups[:-1]) + ['unknown']
        #plt.xticks(range(len(unique_groups)), labels_fixed)
        if len(ax.get_yticks()) < 10:
            tick_interval = (ax.get_yticks()[-1] - ax.get_yticks()[0]) / (len(ax.get_yticks()) - 1)
            new_ticks = [ax.get_yticks()[0] + i*tick_interval/2 for i in range(2*len(ax.get_yticks())-1)]
            ax.set_yticks(new_ticks)
        plt.axhline(y=np.mean(diff_df[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        # count the number of data points in each group
        counts = diff_df[group_var].value_counts()
        # create a dictionary to map the counts to the group names
        count_dict = dict(zip(counts.index, counts.values))
        # add the counts to the x-tick labels
        if group_var == 'birth_year':
            diff_unique_groups = ['unknown' if x == 2023 else x for x in diff_unique_groups]
            count_dict['unknown'] = count_dict.pop(2023)
        xtick_labels = [f'{name}\n({count_dict[name]})' for name in diff_unique_groups]
        ax.set_xticklabels(xtick_labels)
        plt.legend()
        if counter == 15:
            plt.show(block = True)
        else:
            plt.show(block = False)
    # save the figure
    fig_name = 'violin_swarm_difference_' + str(counter) + '.png'  # change the file name as per your preference
    plt.savefig(save_path + fig_name)

'''
    if  f_result_diff.pvalue < 0.05:
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
        #ax = sns.boxplot(data=new_df, x=group_var, y=hr_var, order=unique_groups)
        ax = sns.swarmplot(data=duplicates, x=group_var, y=hr_var, order=unique_groups, alpha = 0.65)
        # add pointplot with average values
        ax = sns.pointplot(data=duplicates, x=group_var, y=hr_var, color="black", markers="d", scale=1.5, join=False, ci=None, size=10, order=unique_groups, linewidth=2, label='Group Average')
        if hr_var == 'RMSSD' or hr_var =='SDNN' or hr_var =='AVNN':
            ax.set(xlabel ='year', ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        elif hr_var == 'HR':
            ax.set(xlabel ='year', ylabel = hr_var + "(bpm)", title = hr_var +' by '+group_var)
        elif hr_var == 'PNN50':
            ax.set(xlabel ='year', ylabel = hr_var + "(percentage)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel ='year', ylabel = hr_var, title = hr_var +' by '+group_var)
        if group_var == 'gender':
            ax.set(xlabel='Gender')
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
        if len(ax.get_yticks()) < 10:
            tick_interval = (ax.get_yticks()[-1] - ax.get_yticks()[0]) / (len(ax.get_yticks()) - 1)
            new_ticks = [ax.get_yticks()[0] + i*tick_interval/2 for i in range(2*len(ax.get_yticks())-1)]
            ax.set_yticks(new_ticks)
        plt.axhline(y=np.mean(duplicates[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        # count the number of data points in each group
        counts = duplicates[group_var].value_counts()
        # create a dictionary to map the counts to the group names
        count_dict = dict(zip(counts.index, counts.values))
        # add the counts to the x-tick labels
        xtick_labels = [f'{name}\n({count_dict[name]})' for name in unique_groups]
        ax.set_xticklabels(xtick_labels)
        plt.legend()
        if counter == 15:
            plt.show()
        else:
            plt.show(block = False)
        
    else:
        print('Unable to reject hypothesis')
        
        ax = sns.swarmplot(data=duplicates, x=group_var, y=hr_var, order=unique_groups, alpha = 0.65)
        # add pointplot with average values
        ax = sns.pointplot(data=duplicates, x=group_var, y=hr_var, color="black", markers="d", scale=1.5, join=False, ci=None, size=10, order=unique_groups, linewidth=2, label='Group Average')
        if hr_var == 'RMSSD' or hr_var =='SDNN' or hr_var =='AVNN':
            ax.set(xlabel ='year', ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        elif hr_var == 'HR':
            ax.set(xlabel ='year', ylabel = hr_var + "(bpm)", title = hr_var +' by '+group_var)
        elif hr_var == 'PNN50':
            ax.set(xlabel ='year', ylabel = hr_var + "(percentage)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel ='year', ylabel = hr_var, title = hr_var +' by '+group_var)
        if group_var == 'gender':
            ax.set(xlabel='Gender')
        plt.title(hr_var +' by date of recording')
        ax.text(0.89, 1.006, "n = " + str(len(duplicates)) + " subjects", transform=ax.transAxes, fontsize=9)
        #labels_fixed = list(unique_groups[:-1]) + ['unknown']
        #plt.xticks(range(len(unique_groups)), labels_fixed)
        if len(ax.get_yticks()) < 10:
            tick_interval = (ax.get_yticks()[-1] - ax.get_yticks()[0]) / (len(ax.get_yticks()) - 1)
            new_ticks = [ax.get_yticks()[0] + i*tick_interval/2 for i in range(2*len(ax.get_yticks())-1)]
            ax.set_yticks(new_ticks)
        plt.axhline(y=np.mean(duplicates[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        # count the number of data points in each group
        counts = duplicates[group_var].value_counts()
        # create a dictionary to map the counts to the group names
        count_dict = dict(zip(counts.index, counts.values))
        # add the counts to the x-tick labels
        xtick_labels = [f'{name}\n({count_dict[name]})' for name in unique_groups]
        ax.set_xticklabels(xtick_labels)
        plt.legend()
        if counter == 15:
            plt.show(block = True)
        else:
            plt.show(block = False)
    # save the figure
    fig_name = 'swarm_plotting_' + str(counter) + '.png'  # change the file name as per your preference
    plt.savefig(save_path + fig_name)
    '''



if __name__ == '__main__':

    all_hrv = '/Users/felixbrener/Documents/ARC/all_hrv.xlsx'
    hrv_list = ['HR','AVNN', 'SDNN', 'RMSSD','PNN50']
    group_list = ['Sport','birth_year','gender']
    #group_var = 'date_year'
    counter = 0

    #cleans up the df so that we can finds all duplicate subjects in the data
    all_hrv_df = find_duplicates(all_hrv)
    '''
    for hrv in hrv_list:
        hr_var = hrv

        #finds all unique values in demographics data columns
        unique_groups, grouped_data = group_data_by_variable(all_hrv_df, group_var, hr_var)
        #performs one way ANOVa and tukey's HSD and then created box plots showing significant differences
        stats_and_plotting(unique_groups,grouped_data, all_hrv_df, hr_var,group_var, counter)
        #stats_and_plotting(all_hrv_df, hr_var,group_var, counter)
        counter = counter + 1

        #stats_and_plotting calls group_data_by_variable
        #group_data_by_variable calls add_demographics_data
        #add_demographics_data calls read_demographics_data and read_hrv_data 
        #group_data_by_variable()
    ''' 
    
        
    for group_var in group_list:
        for hrv in hrv_list:
            hr_var = hrv

            #finds all unique values in demographics data columns
            unique_groups, grouped_data = group_data_by_variable(all_hrv_df, group_var, hr_var)
            #performs one way ANOVa and tukey's HSD and then created box plots showing significant differences
            stats_and_plotting(unique_groups,grouped_data, all_hrv_df, hr_var,group_var, counter)
            #stats_and_plotting(all_hrv_df, hr_var,group_var, counter)
            counter = counter + 1

            #stats_and_plotting calls group_data_by_variable
            #group_data_by_variable calls add_demographics_data
            #add_demographics_data calls read_demographics_data and read_hrv_data 
            #group_data_by_variable()
        
