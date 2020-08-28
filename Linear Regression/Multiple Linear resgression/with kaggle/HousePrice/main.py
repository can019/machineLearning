import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

data = pd.read_csv('kc_house_data.csv')
newdata = data[['bedrooms','bathrooms','sqft_living','sqft_lot', 'floors', 'waterfront','view','price']]
numpyData = np.array()