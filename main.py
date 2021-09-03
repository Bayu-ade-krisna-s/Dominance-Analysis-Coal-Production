# Data Preparation
import pandas as pd
data = pd.read_csv('coal_production.csv')

# Define predictor variables and target variable
x = data[['Rain','Slippery','Rainfall']]
y = data[['Prod_coal']]

# Data Modelling Using Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lr = LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
model = lr.fit(x_train,y_train)
print(model.intercept_)
print(model.coef_)

# Dominance Analysis
from dominance_analysis import Dominance
dominance_regression = Dominance(data=data.drop(['Tanggal','Prod_OB'],axis=1),target='Prod_coal',objective=1)
incr_variable_rsquare = dominance_regression.incremental_rsquare()
dominance_regression.plot_incremental_rsquare()
