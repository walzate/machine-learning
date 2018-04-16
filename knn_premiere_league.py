import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glob
import os

from sklearn.neighbors import KNeighborsClassifier

# based on https://hub.coursera-notebooks.org/user/smppgbjoljzaooewjwbuoj/notebooks/Module%201.ipynb
# data from http://football-data.co.uk/englandm.php
knn = None
lookup_home_team = None
lookup_ftr = None
file_name = 'total.csv'
data_path = 'seasons/'
final_prefix = 'final_'
withYear = False

def merge_files():
    path = data_path
    allFiles = glob.glob(path + "/E0_*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        print(os.path.basename(file_))
        df = pd.read_csv(file_,index_col=None, header=0, error_bad_lines=False, skip_blank_lines = True)
        df['Date'] =  pd.to_datetime(df['Date'])
        df['HomeTeam'] = df['HomeTeam'].replace('', np.nan)
        df.dropna(subset=['HomeTeam'], inplace=True)
        list_.append(df)
    frame = pd.concat(list_)
    frame.to_csv(data_path+'total.csv')


def preprocess():
    global file_name
    global data_path
    global final_prefix

    # Loading the file
    matches = pd.read_csv(data_path+file_name)

    matches = matches[['Date', 'HomeTeam', 'AwayTeam', 'FTR']]

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
    
    matches['HomeTeam'] = matches['HomeTeam'].str.replace('Middlesboro','Middlesbrough')
    matches['AwayTeam'] = matches['AwayTeam'].str.replace('Middlesboro','Middlesbrough')

    matches = matches[matches.FTRcat != -1]

    print(matches.dtypes)
    print(matches.head())
    matches.to_csv(data_path + final_prefix + file_name)

def fit(k):
    global knn
    global lookup_home_team
    global lookup_ftr
    global lookup_away_team
    global file_name
    global data_path
    global final_prefix
    global withYear


    # Loading the file
    matches = pd.read_csv(data_path + final_prefix + file_name)

    # X features, y labels
    if(withYear):
        X = matches[['HomeTeamcat', 'AwayTeamcat', 'MatchYear']]
    else:
        X = matches[['HomeTeamcat', 'AwayTeamcat']]
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
    global withYear
    yearToEvaluate = 2018

    if(withYear):
        match_prediction = knn.predict([[lookup_home_team[home], lookup_home_team[away], yearToEvaluate]])
    else:
        match_prediction = knn.predict([[lookup_home_team[home], lookup_home_team[away]]])
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
    maxNumberOfGames = 25 

    for k in range(1, maxNumberOfGames):
        foundAccuracy = fit(k)
        if(foundAccuracy > bestAccuracy):
            bestAccuracy = foundAccuracy
            bestK = k
    print('Best k: ',bestK, 'Best accuracy: ', bestAccuracy)
    return bestK
            

# Without Year: Best k:  19 Best accuracy:  0.46150592216582065
# With Year:    Best k:  19 Best accuracy:  0.4483925549915397

#merge_files()            
#preprocess()
#findBestK()
#fit(findBestK())
fit(19)
predictions()

