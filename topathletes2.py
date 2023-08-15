#topathletes2
import pandas as pd
import numpy as np


def get_top_names(df, sport, hrv_list):
    # Drop rows with empty cells in the name column
    df = df.loc[df['name'] != 'No_Name']

    #df = df[df['name'] != 'No_Name']
    df.loc[:, 'name'] = df['name'].str.lower().str.replace(' ', '')
    print('hi')
    #df['name'] = df['name'].str.lower().str.replace(' ', '')
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
    columns_of_interest = ['name', 'HR', 'RMSSD', 'AVNN', 'SDNN', 'PNN50']
    df_selected = df[columns_of_interest]

    # Normalize the heart rate variable columns
    df_normalized = (df_selected.iloc[:, 1:] - df_selected.iloc[:, 1:].min()) / (df_selected.iloc[:, 1:].max() - df_selected.iloc[:, 1:].min())

    # Calculate the rank of each name within each heart rate variable
    rank_df = df_normalized.rank(ascending=False)

    # Convert the rank to a percentage score
    percentage_df = (rank_df / df_normalized.shape[0]) * 100
    #inverting the HR percentages because lower HR is better
    percentage_df['HR'] = (percentage_df['HR'] - 100).abs()

    # Calculate the average across the numerical data columns
    percentage_df['hrv_percentile'] = percentage_df.iloc[:, 1:].mean(axis=1)

    # Drop the original numerical data columns
    percentage_df =  percentage_df.drop(percentage_df.columns[1:6], axis=1)

    # Calculate the threshold value for the top 5th percentile
    threshold = df['percentile'].quantile(0.95)

    # Filter the DataFrame to include only names in the top 5th percentile
    top_names = df[df['percentile'] >= threshold]['name'].tolist()

    return top_names


def get1(df, sport, hrv_list):
    # Drop rows with empty cells in the name column
    df = df[df['name'] != 'No_Name']
    df['name'] = df['name'].str.lower().str.replace(' ', '')
    df['birthday'] = pd.to_datetime(df['birthday'])
    df['date'] = pd.to_datetime(df['date'])
    # Calculate the age in years
    age = (df['date'] - df['birthday']).dt.total_seconds() / (365.25 * 24 * 3600)
    df['age'] = age
    df['age'] = df['age'].apply(np.floor).astype(int)
    df.loc[df['age'] < 5, 'age'] = 'unknown'
    df['age'] = df['age'].astype(str)
    df['email'] = df['email'].astype(str)
    df1 = pd.DataFrame()
    df_email = pd.DataFrame()
    sport_df = df[df['Sport'] == sport]
    string_dict = {}

    for hrv in hrv_list:
        result = []
        if hrv == 'HR':
            sorted_df = sport_df.sort_values(hrv)
            sorted_df = sorted_df[sorted_df[hrv] > 0]
            sorted_df = sorted_df.drop_duplicates(subset='name', keep='last')
        else:
            sorted_df = sorted_df.drop_duplicates(subset='name', keep='last')
            sorted_df = sport_df.sort_values(hrv, ascending=False)
        num_students = sport_df.shape[0]
        num_lowest = int(num_students * 0.1)
        result.extend(sorted_df['email'].iloc[:num_lowest])
        # Create a DataFrame from the list
        new_column = pd.DataFrame({sport + ': ' + hrv: result})
        df_email = pd.concat([df_email, new_column], axis=1)
        for index, row in sorted_df.iterrows():
            hrv_string = str(row[hrv])
            string = 'email: ' + row['email'] + ', name: ' + row['name'] + ', score: ' + hrv_string + ', age: ' + row['age']
            email = row['email']
            string_dict[email] = string
    
        new_column[sport + ': ' + hrv] = new_column[sport + ': ' + hrv].replace(string_dict)
        
        # Concatenate the new column to the existing DataFrame
        df1 = pd.concat([df1, new_column], axis=1)
    return df1, df_email, string_dict, sorted_df

def email_ranking1(sorted_df, df_email, sport):
    # Initialize dictionaries to store the information
    email_counts = {}
    email_avg_indices = {}

    # Iterate over the columns
    for column in df_email.columns:
        #reverse the order of the columns for the weighted avg
        #df_email = df_email.iloc[::-1]
        # Get the unique emails and their indices in the current column
        unique_emails, indices, counts = np.unique(df_email[column], return_index=True, return_counts=True)
        #go to each metric and get the percentage score (how high did they rank within their team/HR on each of the valued scored)
        #weighted equally for 20% of each of the scores (each musltiplied by 0.2)
        #find the top 5% highest scorign athletes based on those athletes who have been weighted(add them up)
        #Update the dictionaries with the information
        for email, index, count in zip(unique_emails, indices, counts):
            email_counts[email] = email_counts.get(email, 0) + count
            if email not in email_avg_indices:
                email_avg_indices[email] = 0
            email_avg_indices[email] += index * count

    # Calculate the average indices
    for email in email_avg_indices:
        email_avg_indices[email] /= email_counts[email]

    # Create a DataFrame to store the results
    result_df = pd.DataFrame({
        'Email': list(email_counts.keys()),
        'Count': list(email_counts.values()),
        'Average Index': list(email_avg_indices.values())
    })

    # Calculate the weighted average index
    result_df['Weighted Average'] = result_df['Average Index'] * result_df['Count'] / df_email.shape[1]

    # Sort the DataFrame by Count in descending order
    result_df = result_df.sort_values('Weighted Average', ascending=False).reset_index(drop=True)
    # Keep only the first column
    result_df = result_df.iloc[:, 0:1]
    print('columns are:',result_df.columns)
    for index, row in sorted_df.iterrows():
        string = 'email: ' + row['email'] + ', name: ' + row['name'] + ', age: ' + row['age']
        email = row['email']
        string_dict[email] = string
    
    result_df['Email'] = result_df['Email'].replace(string_dict)
    result_df = result_df.rename(columns={'Email': sport})
    return result_df




# Call the function
hrv_list = ['HR','AVNN','SDNN','RMSSD','PNN50']
df = pd.read_excel('/Users/felixbrener/df_with_email.xlsx')
sports = df['Sport'].unique()
df_with_top_athletes = pd.DataFrame()
df = df.drop(df.columns[0], axis=1)
for sport in sports:
    df = df[df['Sport'] == sport].reset_index(drop=True)
    #df1, df_email, string_dict, sorted_df, = get_top_names(df, sport, hrv_list)
    top_names = get_top_names(df, sport, hrv_list)
    result_df = email_ranking1(sorted_df, df_email, sport)
    df_with_top_athletes  = pd.concat([df_with_top_athletes, result_df], axis=1)
#df_with_top_athletes.to_excel('top_athletes4.xlsx', index=False)
