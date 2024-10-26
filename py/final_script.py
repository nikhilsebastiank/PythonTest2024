#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze > requirements.txt')


# In[2]:


## Packages:
import pandas as pd
import numpy as np
import nltk
import re
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set input and output paths: 
data_path = '~/Dropbox/Projects/WorldBank-Trade/PythonRStataTest2024/PythonTest/data'
out_path = '~/Dropbox/Projects/WorldBank-Trade/PythonRStataTest2024/PythonTest/output'

# Read the dataset and sort by country and company names:
foreign_names = pd.read_csv(data_path + '/ForeignNames_2019_2020.csv', )
foreign_names

iso_codes = pd.read_csv(data_path + '/Country_Name_ISO3.csv')
merged_df = pd.merge(foreign_names, iso_codes, left_on='foreigncountry_cleaned', right_on='country_name', how='left')


# In[3]:


merged_df


# In[4]:


## Logic for the cleaning algorithm that creates the unique ids:
#1. Preprocess the strings - convert to lower case, remove white spaces, non alphabetic characters, stopwords (ltd, limited, corp, etc.)
#2. Compute distance ratios between each pair after sorting 
#3. If similarity >= 80 and < 100 (arbitrarily chosen based on the Teaboard Ltd. example), the two companies are the same.
#4. Add a column to say whether the name has to be changed with values Yes/No.
#5. If Yes, look for the nearest No value (after sorting) and use the firm name from that row. 
#6. Assign unique ids:


# In[5]:


# Functions:
# For Pre-processing text:

def pre_process(text):
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords:
    stop_words = ['limited', 'ltd', 'corporation', 'corp', 'co.', 'industries', 'private', 'public']
    # Split the string into words based on stopwords:
    for stopword in stop_words:
        text = text.replace(stopword, "")
    # Remove white spaces:
    text = ''.join(text.split())
    return(text)

# For writing the unique ids:
def comp_similar2(df):
    # Drop NaNs and sort the DataFrame
    df.dropna(inplace=True)
    df = df.sort_values(by=["foreigncountry_cleaned", "foreign"])[["foreigncountry_cleaned", "foreign", "shpmtyear", "country_iso3"]]
    
    # Pre-process using the pre-defined function above
    df['foreign_processed'] = df['foreign'].apply(pre_process) 
    
    # Shifting the entire column 'foreign_processed' by 1 and adding it to column 'sequential'
    df['sequential'] = df['foreign_processed'].shift(-1)
    
    # Applying the fuzz ratio for each pair post the shift
    df['dist_ratio'] = df.apply(
    lambda x: fuzz.ratio(x['foreign_processed'], x['sequential']) if pd.notnull(x['sequential']) else None, axis=1)
    
    #Using the threshold to find indices of values that needs to be changed
    df['change'] = np.where((df['dist_ratio'] >= 80) &  (df['dist_ratio'] < 100), 'yes', 'no')
    var_yes = np.where(df['change']=='yes')
    
    # Initialize a cleaned_name column and assign the clean
    df['cleaned_name'] = df['foreign']
    for i in var_yes:
        df['cleaned_name'].iloc[i] = df['foreign'].iloc[i-1] # only works with even no. of firms
    
    # Create a unique id for each firm:
    df['cleaned_ID'] = pd.factorize(df['cleaned_name'])[0]
    # Convert the ids to strings and prefix it with the iso3 code:
    df['cleaned_ID'] = df['cleaned_ID'].astype(str)
    df['cleaned_ID'] = df['country_iso3'] + df['cleaned_ID']
    return df


# In[6]:


# Split data into training and test data:
train_data, test_data = train_test_split(merged_df, test_size=0.3, random_state = 40)


# In[7]:


# Apply the initial cleaning algorithm to the test data:
train_data = comp_similar2(train_data)


# In[8]:


## Random Forest Classifier: Unfortunately, my computer runs out of memory when I run the following block, 
## leading it to crash. 

# Using 'foreign_processed' as the input and 'cleaned_name' as the target for prediction
#X = train_data['foreign_processed']
#y = train_data['cleaned_name']

# Split the sampled data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 40)

# Apply TF-IDF vectorization with 500 features
#vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=500)
#X_train_tfidf = vectorizer.fit_transform(X_train)
#X_test_tfidf = vectorizer.transform(X_test)

# Re-initialize and train the Random Forest Classifier with the smaller subset
#rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#rf_model.fit(X_train_tfidf, y_train)

# Predict on the test data
#y_pred = rf_model.predict(X_test_tfidf)

# Calculate and display the accuracy and classification report
#accuracy = accuracy_score(y_test, y_pred)
#classification_rep = classification_report(y_test, y_pred, zero_division=1)

#accuracy, classification_rep


# In[9]:


# Output files in part1:
cleaned = comp_similar2(merged_df)


# In[10]:


# Output 1
columns_to_keep = ['foreign', 'foreigncountry_cleaned', 'shpmtyear', 'cleaned_ID', 'cleaned_name']
cleaned_subset = cleaned[columns_to_keep]
cleaned_subset.to_csv(out_path + '/output_nikhil_1.csv')


# In[11]:


# Output1 Changed Names:
cleaned_changed = cleaned[cleaned['change']=='yes'] ## Only firms whos names have been changed, needs further checking
cleaned_changed = cleaned_changed[['foreign', 'cleaned_name', 'cleaned_ID']]
cleaned_changed.to_csv(out_path + '/output_nikhil_1_changed.csv')


# In[12]:


## Part 2:
foreign_names_2021 = pd.read_csv(data_path + '/ForeignNames_2021.csv', )


# In[13]:


# Join the two data frames:
merged_df2 = pd.merge(cleaned, foreign_names_2021, on = ['foreign', 'foreigncountry_cleaned', 'shpmtyear'], how='outer')


# In[14]:


merged_df2
cleaned2 = comp_similar2(merged_df2)


# In[15]:


cleaned2
cleaned_subset2 = cleaned2[columns_to_keep]
cleaned_subset2.to_csv(out_path + "/output_nikhil_2.csv")

