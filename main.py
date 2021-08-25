import pandas as pd
import numpy as np
import logging

from pandas.core.tools.datetimes import DatetimeScalar
from fuzzycmeans.fuzzy_clustering import FCM
from fuzzycmeans.visualization import draw_model_2d
from sklearn import preprocessing

dataset = pd.read_csv("Variables PDM y Molino Proyecto Hermes 2021 - Fuzzy Rules Interview 3 - Answers Delta.csv") 
#Importing the airlines data.

dataset1 = dataset.copy()
# Making a copy so that the original data remains unaffected.

dataset1 = dataset1[["Caudal de M.P. [kg] (19500)", "Caudal de M.P. [kg] (19500)"]]
# Selecting only first 500 rows for faster computation

dataset1_std = preprocessing.scale(dataset1)
# Standardizingthe data to scale it between the upper and lower limit of 1 and 0.

dataset1_std = pd.DataFrame(dataset1_std)

fcm = FCM(n_clusters=5)
# Defining k=5.

fcm.set_logger(tostdout=False)
# Telling the package class to stop the unnecessary output.

fcm.fit(dataset1_std)
# Training on data.

predicted_membership = fcm.predict(np.array(dataset1_std))
# Testing on same data.

draw_model_2d(fcm, data=np.array(dataset1_std), membership=predicted_membership)
# Visualizing data.
