
# importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# read the train and test dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.head())

# shape of the dataset
print('\nShape of training data :',train_data.shape)
print('\nShape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Item_Outlet_Sales

# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y = train_data['Item_Outlet_Sales']

# seperate the independent and target variable on training data
test_x = test_data.drop(columns=['Item_Outlet_Sales'],axis=1)
test_y = test_data['Item_Outlet_Sales']

'''
Create the object of the Linear Regression model
You can also add other parameters and test your code here
Some parameters are : fit_intercept and normalize
Documentation of sklearn LinearRegression: 

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

 '''
model = LinearRegression()

# fit the model with the training data
model.fit(train_x,train_y)

# coefficeints of the trained model
print('\nCoefficient of model :', model.coef_)

# intercept of the model
print('\nIntercept of model',model.intercept_)

# Feature Importance (Coefficients)
feature_importance = pd.Series(model.coef_, index=train_x.columns)
plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind='barh', color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()

# predict the target on the test dataset
predict_train = model.predict(train_x)
print('\nItem_Outlet_Sales on training data',predict_train) 

# Root Mean Squared Error on training dataset
rmse_train = mean_squared_error(train_y,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

# Training Data: Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.scatter(train_y, predict_train, alpha=0.5, label='Train Data', color='blue')
plt.plot([min(train_y), max(train_y)], [min(train_y), max(train_y)], color='red', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Train Data)')
plt.legend()
plt.show()

# predict the target on the testing dataset
predict_test = model.predict(test_x)
print('\nItem_Outlet_Sales on test data',predict_test) 

# Root Mean Squared Error on testing dataset
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
print('\nRMSE on test dataset : ', rmse_test)

# Testing Data: Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.scatter(test_y, predict_test, alpha=0.5, label='Test Data', color='green')
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Test Data)')
plt.legend()
plt.show()

# Residuals on Training Data
residuals_train = train_y - predict_train
plt.figure(figsize=(10, 6))
sns.histplot(residuals_train, kde=True, color='blue')
plt.title('Residuals on Training Data')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Residuals on Testing Data
residuals_test = test_y - predict_test
plt.figure(figsize=(10, 6))
sns.histplot(residuals_test, kde=True, color='green')
plt.title('Residuals on Testing Data')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

