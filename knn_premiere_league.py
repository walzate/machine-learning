import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# based on https://hub.coursera-notebooks.org/user/smppgbjoljzaooewjwbuoj/notebooks/Module%201.ipynb

# Loading the file
matches = pd.read_csv('premiere_league_total_1993-2015_preprocesado_solo_caracteristicas.csv')

# Preprocessing
matches['HomeTeam'] = matches['HomeTeam'].str.replace('\'', '')
matches['HomeTeam'] = matches['HomeTeam'].str.replace(' ', '')
matches['AwayTeam'] = matches['AwayTeam'].str.replace('\'', '')
matches['AwayTeam'] = matches['AwayTeam'].str.replace(' ', '')

matches['HomeTeam'] = matches['HomeTeam'].astype('category')
matches['HomeTeamcat'] = matches['HomeTeam'].cat.codes
matches['AwayTeam'] = matches['AwayTeam'].astype('category')
matches['AwayTeamcat'] = matches['AwayTeam'].cat.codes
matches['FTR'] = matches['FTR'].astype('category')
matches['FTRcat'] = matches['FTR'].cat.codes

print(matches.dtypes)
print(matches.head())

# X features, y labels
X = matches[['HomeTeamcat', 'AwayTeamcat']]
y = matches['FTRcat']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create classifier object
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
print(knn.score(X_test, y_test))

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_ftr = dict(zip(matches.FTRcat.unique(),matches.FTR.unique()))   
print(lookup_ftr)
lookup_home_team = dict(zip(matches.HomeTeam.unique(), matches.HomeTeamcat.unique()))   
print(lookup_home_team)
lookup_away_team = dict(zip(matches.AwayTeam.unique(), matches.AwayTeamcat.unique()))   
print(lookup_away_team)


#Predictions
def predict(home, away):
    match_prediction = knn.predict([[lookup_home_team[home], lookup_home_team[away]]])
    print(home,' vs ',away, ': ',lookup_ftr[match_prediction[0]])

predict('Arsenal', 'ManUnited')
predict('ManCity', 'Liverpool')
predict('ManCity', 'Arsenal')
predict('Arsenal', 'Wigan')
predict('ManCity', 'ManUnited')
predict('ManCity', 'QPR')
predict('Chelsea', 'Newcastle')
predict('Arsenal', 'Sunderland')

