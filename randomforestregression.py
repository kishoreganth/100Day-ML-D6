#importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# importing the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#splitting into train and test set 

#feature scaling 

# fitting into the Random FOrest Regression
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x, y)


# predicting the new result
y_pred = regressor.predict(6.5)

#visualising the results from the Random Forest REgressor 
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid, regressor.predict(x_grid),color='blue')
plt.title('Truth or Bluff(RandomForest Regresor)')
plt.xlabel('position salaries')
plt.ylabel('level')
plt.show()
