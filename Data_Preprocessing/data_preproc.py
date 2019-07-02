#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#data preprocessing

# import des lib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import du dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# gestion des données manquantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# gestion des variables catégoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(n_values='auto', categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# division du dataset entre training set et le test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# feature scaling

from sklearn.prepocessing import StandardScaler

sc = StandardScaler()
