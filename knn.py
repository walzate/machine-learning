import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# based on https://hub.coursera-notebooks.org/user/smppgbjoljzaooewjwbuoj/notebooks/Module%201.ipynb

fruits = pd.read_table('fruit_data_with_colors.txt')

print(fruits.head())

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
print(lookup_fruit_name)

# For this example, we use the mass, width, and height features of each fruit instance
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Create classifier object
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

#Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

#Estimate the accuracy of the classifier on future data, using the test data
print(knn.score(X_test, y_test))

#Use the trained k-NN classifier model to classify new, previously unseen objects

# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
print(lookup_fruit_name[fruit_prediction[0]])

# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm
fruit_prediction = knn.predict([[192, 8.4, 7.3]])
print(lookup_fruit_name[fruit_prediction[0]])
