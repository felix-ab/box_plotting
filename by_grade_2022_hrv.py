#everyone but freshman
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

def by_grades_2022(all_hrv):
    all_hrv_df = pd.read_excel(all_hrv)
    #all_hrv_df[['quality', 'HR', 'AVNN', 'SDNN','RMSSD','PNN50']] = all_hrv_df[['quality', 'HR', 'AVNN', 'SDNN','RMSSD','PNN50']].apply(pd.to_numeric)
    only_freshman = all_hrv_df[(all_hrv_df['date'].dt.year == 2022) & (all_hrv_df['grade'] == 2026)]
    #group the sport column by its unique values and count the occurrences
    counts = only_freshman.groupby('Sport').size()
    # identify 'Sports' values that appear less than 4 times
    mask = only_freshman['Sport'].isin(counts[counts < 3].index)
    # replace those values with 'other'
    only_freshman.loc[mask, 'Sport'] = 'Other'
    #replace "No_Sport" with "other"
    only_freshman.loc[only_freshman['Sport'] == 'No_Sport', 'Sport'] = 'Other'    

    no_freshman = all_hrv_df[(all_hrv_df['date'].dt.year == 2022) & (all_hrv_df['grade'].isin([2025, 2024, 2023]))]
    #group the sport column by its unique values and count the occurrences
    counts = no_freshman.groupby('Sport').size()
    # identify 'Sports' values that appear less than 4 times
    mask = no_freshman['Sport'].isin(counts[counts < 3].index)
    # replace those values with 'other'
    no_freshman.loc[mask, 'Sport'] = 'Other'

    return only_freshman, no_freshman

def create_pie_chart(all_hrv):
    all_hrv_df = pd.read_excel(all_hrv)
    # extract year and create new column
    all_hrv_df['date_year'] = all_hrv_df['date'].dt.year
    all_hrv_df['name'] = all_hrv_df['name'].str.lower().str.replace(' ', '')
    # Group by name and year, and count the number of entries
    counts = all_hrv_df.groupby(['name', 'date_year']).size().unstack(fill_value=0)
    counts = pd.DataFrame(counts)
    counts.index.name = 'name'
    counts.columns.name = 'date_year'
    # Reset index to convert 'name' from index to a regular column
    counts = counts.reset_index()
    # Rename the columns to the desired format
    # count the number of rows that meet the specified conditions
    num_2021_gt_0_and_2022_eq_0 = len(counts[(counts[2021] > 0) & (counts[2022] == 0)])
    num_2021_eq_0_and_2022_gt_0 = len(counts[(counts[2021] == 0) & (counts[2022] > 0)])
    num_2021_and_2022_gt_0 = len(counts[(counts[2021] > 0) & (counts[2022] > 0)])
    # Print the results
    print(f"Number of rows with 2021 > 0 and 2022 = 0: {num_2021_gt_0_and_2022_eq_0}")
    print(f"Number of rows with 2021 = 0 and 2022 > 0: {num_2021_eq_0_and_2022_gt_0}")
    print(f"Number of rows with 2021 > 0 and 2022 > 0: {num_2021_and_2022_gt_0}")
    # numbers of subjects in each category
    numbers = [num_2021_gt_0_and_2022_eq_0, num_2021_eq_0_and_2022_gt_0, num_2021_and_2022_gt_0]
    # Labels for each number
    labels = ['Only 2021\n'+'('+str(num_2021_gt_0_and_2022_eq_0)+')', 'Only 2022\n'+'('+str(num_2021_eq_0_and_2022_gt_0)+')', 'Both years\n'+'('+str(num_2021_and_2022_gt_0)+')']
    # Creating the pie chart
    fig0, ax0 = plt.subplots(figsize=(10,10))
    ax0.pie(numbers, labels=labels, autopct='%1.1f%%')
    # Adding a title 
    ax0.set_title('Subjects by Data Collection Year')
    fig0.savefig('years_piechart.png')
    #By gender
    fig1, ax1 = plt.subplots(figsize=(10,10))
    gender_counts = all_hrv_df['gender'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(10,10))
    gender_counts = gender_counts.rename(index={'female':'Female','male':'Male','unknown': 'N.A'})
    ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',)
    ax1.set_title('Gender Distribution')
    fig1.savefig('gender_piechart.png')
    # Sport pie chart
    sport_counts = all_hrv_df['Sport'].value_counts()
    all_hrv_df['Sport_reduced'] = all_hrv_df['Sport'].apply(lambda x: x if sport_counts[x] >= 15 else 'Other')
    counts_reduced = all_hrv_df['Sport_reduced'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(10,10))
    ax2.pie(counts_reduced, labels=counts_reduced.index, autopct='%1.1f%%', labeldistance=1.1, pctdistance=0.8)
    ax2.text(-0.65, 1.37, '* Sports with less than 15 entries are in the "Other" category', fontsize=10, color='grey')
    ax2.set_title('Sports Distribution')
    plt.savefig('sports_piechart3.png')
    plt.show()

def group_data_by_variable(DataFrame, group_var, hr_var):
    grouped_data = []
    unique_groups = np.unique(DataFrame[group_var])
    if group_var == 'Sport':
        idx = np.where(unique_groups == 'Other')[0][0]
        unique_groups = np.concatenate((np.delete(unique_groups, idx), ['Other']))
        sport_order = {'Boys Basketball': 0, 'Girls Basketball': 1,'Boys Soccer': 2,
        'Girls Soccer': 3,'Cross Country': 4, 'Field Hockey': 5,'Football': 6,'Ice Hockey': 7,'Squash': 8,
        'Tennis': 9,'Volleyball': 10,'Water Polo': 11,'Basketball': 12, 'No_Sport': 13, 'Other': 14}
        unique_groups = sorted(unique_groups, key=lambda s: sport_order[s])
    for group in unique_groups:
        grouped_data.append(DataFrame[DataFrame[group_var] == group][hr_var])

    return unique_groups, grouped_data

def stats_and_plotting(unique_groups,grouped_data,DataFrame,hr_var,group_var, counter):
    if str(max(DataFrame['grade'])) == '2025':
        grades = 'no_freshman_'
        title_grades = 'No Freshman'
    else:
        grades = 'only_freshman_'
        title_grades = 'Only Freshman'
    save_path = '/Users/felixbrener/Documents/violin_swarm/'
    f_result= stats.f_oneway(*grouped_data)
    plt.figure(counter, figsize=(13, 8))
    counter = counter + 1
    #removing erronoius measurements
    #double check if there's a second zero for rmssd
    if hr_var == 'PNN50':
        DataFrame = DataFrame[DataFrame[hr_var] >= 0]
        minimumval = ' values less than\nzero '
    elif hr_var == 'RMSSD':
        DataFrame = DataFrame[DataFrame['AVNN'] > 0]
        minimumval = ' values less than\nzero '
    else:
        DataFrame = DataFrame[DataFrame[hr_var] > 0]
        minimumval = ' values of zero or\nless '
    #check again for values that should go in the other category
    #group the sport column by its unique values and count the occurrences
    counts = DataFrame.groupby('Sport').size()
    # identify 'Sports' values that appear less than 4 times
    mask = DataFrame['Sport'].isin(counts[counts < 3].index)
    # replace those values with 'other'
    DataFrame.loc[mask, 'Sport'] = 'Other'
    #if statement to check of all the unique grousp still exist in the data set
    #reseting unique groups now that more values have been removed
    unique_groups = np.unique(DataFrame[group_var])
    if group_var == 'Sport':
        sport_order = {'Boys Basketball': 0, 'Girls Basketball': 1,'Boys Soccer': 2,
        'Girls Soccer': 3,'Cross Country': 4, 'Field Hockey': 5,'Football': 6,'Ice Hockey': 7,'Squash': 8,
        'Tennis': 9,'Volleyball': 10,'Water Polo': 11,'Basketball': 12, 'No_Sport': 13, 'Other': 14}
        unique_groups = sorted(unique_groups, key=lambda s: sport_order[s])
    if  f_result.pvalue < 0.05:
        #Performing Tukey's HSD test
        tukey_df = posthoc_tukey(DataFrame, val_col=hr_var, group_col=group_var)
        #checking result of posthoc_tukey's
        #comverting the matrix to a non-redundant list of comparisons with the p-value
        remove = np.tril(np.ones(tukey_df.shape), k=0).astype("bool")
        tukey_df[remove] = np.nan
        #removing the lower half and diagonal of the matrix and turning the matrix format into a long dataframe
        molten_df = tukey_df.melt(ignore_index=False).reset_index().dropna()
        #filtering the rows where the value is less than 0.05
        filtered_rows = molten_df[molten_df['value'] < 0.05]
        #creatign swarm plots
        ax = sns.violinplot(data=DataFrame, x=group_var, y=hr_var, order=unique_groups, alpha = 0.5)
        ax = sns.swarmplot(data=DataFrame, x=group_var, y=hr_var, order=unique_groups, alpha = 0.7, color = '#FFFFFF', edgecolor='black')
        #ax = sns.swarmplot(data=DataFrame, x=group_var, y=hr_var, order=unique_groups, alpha = 0.65)
        # add pointplot with average values
        ax = sns.pointplot(data=DataFrame, x=group_var, y=hr_var, color="black", markers="d", scale=1.5, join=False, ci=None, size=10, order=unique_groups, linewidth=2, label='Group Average')
        
        if hr_var == 'RMSSD' or hr_var =='SDNN' or hr_var =='AVNN':
            ax.set(xlabel =group_var, ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        elif hr_var == 'HR':
            ax.set(xlabel =group_var, ylabel = hr_var + "(bpm)", title = hr_var +' by '+group_var)
        elif hr_var == 'PNN50':
            ax.set(xlabel =group_var, ylabel = hr_var + "(percentage)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel =group_var, ylabel = hr_var, title = hr_var +' by '+group_var)
        if group_var == 'gender':
            ax.set(xlabel='Gender')
        ax.text(0, 1.006, "p-value annotation legend:\nNot Significant: p > 5.00e-02\n*: 1.00e-02 < p <= 5.00e-02\n**: 1.00e-03 < p <= 1.00e-02\n***: 1.00e-04 < p <= 1.00e-03\n****: p <= 1.00e-04", transform=ax.transAxes, fontsize=9)
        ax.text(0.8, 1.006, "n = " + str(len(DataFrame)) + " subjects\nAll subjects with "+str(hr_var)+ minimumval + "have been removed from the dataset", transform=ax.transAxes, fontsize=9)
        if len(filtered_rows) > 0:
            pairs = [(i[1]["index"], i[1]["variable"]) for i in filtered_rows.iterrows()]
            p_values = [i[1]["value"] for i in filtered_rows.iterrows()]

            annotator = Annotator(ax, pairs, data=DataFrame, x=group_var, y=hr_var, order=unique_groups)
            annotator.configure(text_format="star", loc="inside")
            annotator.set_pvalues_and_annotate(p_values)
        
        plt.title(hr_var +' by '+ group_var+'\n'+title_grades)
        if len(unique_groups) > 10:
            plt.xticks(rotation=15)
        if group_var == 'birth_year':
            labels_fixed = list(unique_groups[:-1]) + ['unknown']
            plt.xticks(range(len(unique_groups)), labels_fixed)
        # double the number of y-axis tic marks if there are less than 9
        if len(ax.get_yticks()) < 10:
            tick_interval = (ax.get_yticks()[-1] - ax.get_yticks()[0]) / (len(ax.get_yticks()) - 1)
            new_ticks = [ax.get_yticks()[0] + i*tick_interval/2 for i in range(2*len(ax.get_yticks())-1)]
            ax.set_yticks(new_ticks)
        plt.axhline(y=np.mean(DataFrame[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        # count the number of data points in each group
        counts = DataFrame[group_var].value_counts()
        # create a dictionary to map the counts to the group names
        count_dict = dict(zip(counts.index, counts.values))
        # add the counts to the x-tick labels
        if group_var == 'birth_year':
            unique_groups = ['unknown' if x == 2023 else x for x in unique_groups]
            count_dict['unknown'] = count_dict.pop(2023)
        xtick_labels = [f'{name}\n({count_dict[name]})' for name in unique_groups]
        ax.set_xticklabels(xtick_labels)
        plt.legend()
        if counter == 30:
            plt.show()
        else:
            plt.show(block = False)
        
    else:
        print('Unable to reject hypothesis')
        ax = sns.violinplot(data=DataFrame, x=group_var, y=hr_var, order=unique_groups, alpha = 0.5)
        ax = sns.swarmplot(data=DataFrame, x=group_var, y=hr_var, order=unique_groups, alpha = 0.7, color = '#FFFFFF', edgecolor='black')
        #ax = sns.swarmplot(data=DataFrame, x=group_var, y=hr_var, order=unique_groups, alpha = 0.65)
        # add pointplot with average values
        ax = sns.pointplot(data=DataFrame, x=group_var, y=hr_var, color="black", markers="d", scale=1.5, join=False, ci=None, size=10, order=unique_groups, linewidth=2, label='Group Average')
        
        if hr_var == 'RMSSD' or hr_var =='SDNN' or hr_var =='AVNN':
            ax.set(xlabel =group_var, ylabel = hr_var + "(ms)", title = hr_var +' by '+group_var)
        elif hr_var == 'HR':
            ax.set(xlabel =group_var, ylabel = hr_var + "(bpm)", title = hr_var +' by '+group_var)
        elif hr_var == 'PNN50':
            ax.set(xlabel =group_var, ylabel = hr_var + "(percentage)", title = hr_var +' by '+group_var)
        else:
            ax.set(xlabel =group_var, ylabel = hr_var, title = hr_var +' by '+group_var)
        if group_var == 'gender':
            ax.set(xlabel='Gender')
        plt.title(hr_var +' by '+ group_var+'\n'+title_grades)
        ax.text(0.89, 1.006, "n = " + str(len(DataFrame)) + " subjects", transform=ax.transAxes, fontsize=9)
        if len(unique_groups) > 10:
            plt.xticks(rotation=15)
        if group_var == 'birth_year':
            labels_fixed = list(unique_groups[:-1]) + ['unknown']
            plt.xticks(range(len(unique_groups)), labels_fixed)
        # double the number of y-axis tic marks if there are less than 9
        if len(ax.get_yticks()) < 10:
            tick_interval = (ax.get_yticks()[-1] - ax.get_yticks()[0]) / (len(ax.get_yticks()) - 1)
            new_ticks = [ax.get_yticks()[0] + i*tick_interval/2 for i in range(2*len(ax.get_yticks())-1)]
            ax.set_yticks(new_ticks)
        plt.axhline(y=np.mean(DataFrame[hr_var]), color='#00E665', linestyle='--', label='Average '+ hr_var)
        # count the number of data points in each group
        counts = DataFrame[group_var].value_counts()
        # create a dictionary to map the counts to the group names
        count_dict = dict(zip(counts.index, counts.values))
        # add the counts to the x-tick labels
        if group_var == 'birth_year':
            unique_groups = ['unknown' if x == 2023 else x for x in unique_groups]
            count_dict['unknown'] = count_dict.pop(2023)
        xtick_labels = [f'{name}\n({count_dict[name]})' for name in unique_groups]
        ax.set_xticklabels(xtick_labels)
        plt.legend()
        if counter == 30:
            plt.show(block = True)
        else:
            plt.show(block = False)
    # save the figure
    fig_name = "violin_swarm_" + grades + str(counter) + '.png'  # change the file name as per your preference
    plt.savefig(save_path + fig_name)
    return counter + 1

if __name__ == '__main__':

    all_hrv = '/Users/felixbrener/Documents/ARC/all_hrv.xlsx'
    hrv_list = ['HR','AVNN', 'SDNN', 'RMSSD','PNN50']
    group_list = ['Sport','birth_year','gender']
    counter = 0

#create_pie_chart(all_hrv)


for group in group_list:
    group_var = group
    for hrv in hrv_list:
        hr_var = hrv
        only_freshman, no_freshman = by_grades_2022(all_hrv)
        #finds all unique values in demographics data columns for upper grades
        #unique_groups, grouped_data = group_data_by_variable(no_freshman, group_var, hr_var)
        #performs one way ANOVa and tukey's HSD and then created box plots showing significant differences
        #counter = stats_and_plotting(unique_groups,grouped_data, no_freshman, hr_var,group_var, counter)
        #finds all unique values in demographics data columns for freshman
        unique_groups, grouped_data = group_data_by_variable(no_freshman, group_var, hr_var)
        #performs one way ANOVa and tukey's HSD and then created box plots showing significant differences
        counter = stats_and_plotting(unique_groups,grouped_data, no_freshman, hr_var,group_var, counter)
        #stats_and_plotting calls group_data_by_variable
        #group_data_by_variable calls add_demographics_data
        #group_data_by_variable()
