#by_birth_year
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
new_df = pd.read_excel('/Users/felixbrener/Documents/ARC/all_hrv.xlsx')

hrv_list = ['HR','AVNN', 'SDNN', 'RMSSD','PNN50']
save_path = '/Users/felixbrener/Documents/saved_figs2/'


new_df[['quality', 'HR', 'AVNN', 'SDNN','RMSSD','PNN50']] = new_df[['quality', 'HR', 'AVNN', 'SDNN','RMSSD','PNN50']].apply(pd.to_numeric)


counter = 0

for hr_var in hrv_list:

    #removing erronoius measurements
    if hr_var == 'PNN50':
        new_df = new_df[new_df[hr_var] >= 0]
        minimumval = ' values less than\nzero '
    else:
        new_df = new_df[new_df[hr_var] > 0]
        minimumval = ' values of zero or\nless ' 

    new_df.sort_values(by=['date'])
    new_df = new_df.copy()
    new_df.loc[:, 'name'] = new_df['name'].str.lower()
    new_df = new_df.drop_duplicates(subset='name', keep='last')

    # Convert timestamp columns to datetime objects
    new_df['birthday'] = pd.to_datetime(new_df['birthday'])
    new_df['date'] = pd.to_datetime(new_df['date'])

    # Calculate the age in years
    age = (new_df['date'] - new_df['birthday']).dt.total_seconds() / (365.25 * 24 * 3600)
    new_df['age'] = age
    # Remove all rows where age < 10
    new_df = new_df[new_df['age'] >= 10]
    # create a new column with just the year
    new_df['date of recording'] = new_df['date'].dt.year

    plt.figure(counter, figsize=(13, 8))
    # Create scatter plot with regression line
    #sns.regplot(x='age', y=hr_var, data=new_df, color='#9B9CA5')
    # Define color palette
    #palette = sns.color_palette("Greys", len(new_df['birth_year'].unique()))
    palette = mcolors.ListedColormap(['#008DFF', '#00E2C6','#49D600','#F1BB00','#FF7D0A','#FF3F6D'])

    # Create scatter plot with two hue variables
    scatter_plot = sns.scatterplot(x='age', y=hr_var, hue='birth_year', data=new_df, palette=palette,legend=True, alpha=0.9, edgecolor='none')
    #scatter_plot = sns.scatterplot(x='age', y=hr_var, hue='date of recording', data=new_df,marker='x', palette=['black', '#e6e6e6'], alpha=0.8,legend=True, edgecolor=palette)
    #scatter_plot = sns.scatterplot(x='age', y=hr_var, hue='date of recording', size = 100, data=new_df, palette=['blue', 'red'],legend=False, edgecolor='none')

    #scatter_plot = sns.scatterplot(x='age', y=hr_var, hue='birth_year', data=new_df, legend='brief')
    # Calculate the correlation coefficient
    r = np.corrcoef(new_df['age'], new_df[hr_var])[0, 1]
    plt.title('Correlation between ' + hr_var + ' and age', fontweight="bold")
    plt.text(0.9, 1.05, f'Correlation coefficient: {r:.2f}\nn = ' + str(len(new_df)) + " subjects\nAll subjects with "+str(hr_var)+ minimumval + "have been removed from the dataset", ha='center', va='center',fontsize=9, transform=plt.gca().transAxes)
    plt.xlabel('Age (years) at Time of Recording')
    if hr_var == 'RMSSD' or hr_var =='SDNN' or hr_var =='AVNN':
        plt.ylabel(hr_var + "(ms)")
    elif hr_var == 'HR':
        plt.ylabel(hr_var + "(bpm)")
    elif hr_var == 'PNN50':
        plt.ylabel(hr_var + "(percentage)")
    #plt.text(0.6, 0.982, "\nn = " + str(len(new_df)) + " subjects\nAll subjects with "+str(hr_var)+ minimumval + "have been removed from the dataset", fontsize=9)
    fig_name = 'scatter_figure_' + str(counter) + '.png'
    plt.savefig(save_path + fig_name)
    counter = counter + 1
plt.show()

