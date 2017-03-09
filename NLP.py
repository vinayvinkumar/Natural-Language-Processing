# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Reading the data into Pandas dataframe
stories = pd.read_csv('stories.csv')

# Data Cleaning
# Appending the column names to the dataframe and removing the empty rows
stories.columns = ["nul_1","submission_time", "nul_2", "nul-3", "upvotes", "url", "nul-4", "headline"]
stories = stories.dropna()

# Removing 4 columns as it doesn not provide any useful information
stories.drop(stories.columns[[0,2,3,6]], axis=1, inplace=True)

# Tokenization step
tokenized_headlines = []
for item in stories["headline"]:
    tokenized_headlines.append(item.split(" "))
	

# Preprocessing step 	
punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []
for item in tokenized_headlines:
    tokens = []
    for token in item:
        token = token.lower()
        for punc in punctuation:
            token = token.replace(punc, "")
        tokens.append(token)
    clean_tokenized.append(tokens)
	

# Assembling A Matrix (Building a confusion matrix)
unique_tokens = []
single_tokens = []
for tokens in clean_tokenized:
    for token in tokens:
        if token not in single_tokens:
            single_tokens.append(token)
        elif token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)

counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)

# Counting Tokens	
for i, item in enumerate(clean_tokenized):
    for token in item:
        if token in unique_tokens:
            counts.iloc[i][token] += 1
			
# Removing Extraneous Columns (Removing stop words)
word_counts = counts.sum(axis=0)
counts = counts.loc[:,(word_counts >= 5) & (word_counts <= 100)]			

# Cross-Validation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)

# Machine Learning
clf = LinearRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)