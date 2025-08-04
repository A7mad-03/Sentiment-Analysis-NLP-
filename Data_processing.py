import pandas as pd 
import numpy as np
import re


data_path=r"training.1600000.processed.noemoticon.csv"
data = pd.read_csv(data_path,encoding='latin-1',header=None)

data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']


###cheack missing value ||||| cheak and handling outliers  ####
def check_miss_outliers(data):
    print("THE MISSING VALE IN DATA ::")
    print(data.isnull().sum())

    cols_outliers=[] 
    print("THE OUTLIERS ::")
    for cols in data.select_dtypes(include=np.number).columns:
        median_data_cols = data[cols].median()
        Q1 = np.percentile(data[cols].dropna(), 25)
        Q3 = np.percentile(data[cols].dropna(), 75)
        IQR = Q3 - Q1
        Upper = Q3 + (1.5 * IQR)
        Lower = Q1 - (1.5 * IQR)
        col_outliers = []
        for i in data[cols].index: 
            value = data.at[i, cols]
            if value > Upper or value < Lower:
                col_outliers.append(value)
                data.at[i, cols] = median_data_cols
        print("Outliers in column :::", cols, len(col_outliers))



def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


data['clean_text'] = data['text'].apply(clean_text)



### Convert target (0 = negative, 4 = positive) to binary (0, 1) ###

data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)

##Unimportant features##

data_f = data.drop(['ids', 'date', 'flag', 'user', 'text'], axis=1)



# i already have data ##
##data.to_csv('Final data.csv', index=False)


### AFTER APPLY THAT THE DATA IS BALANCING ###
# print (data['target'].value_counts())



