import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

# based on https://hub.coursera-notebooks.org/user/smppgbjoljzaooewjwbuoj/notebooks/Module%201.ipynb
# data from http://football-data.co.uk/englandm.php
knn = None
lookup_home_team = None
lookup_ftr = None

def preprocess():
    # Loading the file
    matches = pd.read_csv('premiere_league_total_1993-2015_preprocesado.csv')

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

    matches['Date'] =  pd.to_datetime(matches['Date'])
    matches['MatchYear'] =  matches['Date'].dt.year

    print(matches.dtypes)
    print(matches.head())
    matches.to_csv('final.csv')

def fit(k):
    global knn
    global lookup_home_team
    global lookup_ftr
    global lookup_away_team

    # Loading the file
    matches = pd.read_csv('final.csv')

    # X features, y labels
    X = matches[['HomeTeamcat', 'AwayTeamcat', 'MatchYear']]
    y = matches['FTRcat']

    # default is 75% / 25% train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    knn = KNeighborsClassifier(n_neighbors = k)

    # Train the classifier (fit the estimator) using the training data
    knn.fit(X_train, y_train)

    # Estimate the accuracy of the classifier on future data, using the test data
    print('Accuracy for k = ',k,' is: ',knn.score(X_test, y_test))

    # create a mapping from match label value to match name to make results easier to interpret
    lookup_ftr = dict(zip(matches.FTRcat.unique(),matches.FTR.unique()))   
    #print(lookup_ftr)
    lookup_home_team = dict(zip(matches.HomeTeam.unique(), matches.HomeTeamcat.unique()))   
    #print(lookup_home_team)
    lookup_away_team = dict(zip(matches.AwayTeam.unique(), matches.AwayTeamcat.unique()))   
    #print(lookup_away_team)
    return knn.score(X_test, y_test)


#Predictions
def predict(home, away):
    global knn
    global lookup_home_team
    global lookup_ftr
    global lookup_away_team

    match_prediction = knn.predict([[lookup_home_team[home], lookup_home_team[away], 2016]])
    print(home,' vs ',away, ': ',lookup_ftr[match_prediction[0]])

def predictions():
    predict('Newcastle', 'Arsenal')
    predict('ManUnited', 'WestBrom')
    predict('Southampton', 'Chelsea')
    predict('Tottenham', 'ManCity')

def findBestK():
    bestK = 0
    bestAccuracy = 0.0
    foundAccuracy = 0.0
    for k in range(1, 45):
        foundAccuracy = fit(k)
        if(foundAccuracy > bestAccuracy):
            bestAccuracy = foundAccuracy
            bestK = k
    print('Best k: ',bestK, 'Best accuracy: ', bestAccuracy)
    return bestK
            

#fit(findBestK())
fit(29)
predictions()

