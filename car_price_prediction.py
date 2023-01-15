#Understanding the Problem Statement - 
#Our job is to predict the price of car based on the following features - 
#Car brand 
#Year of manufacturing 
#Sold price 
#Present price 
#KMS drive 
#Fuel type 
#Seller type 
#Transmission type 
#Owners 

#Workflow for the project 
#1. load car data 
#2. data preprocessing 
#3. train test split 
#4. model training
#5. model used - linear and lasso regression model 
#6. data evaluation - r squared error 

#load libraries 
#data preprocessing and exploration 
import pandas as pd 

#data visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

#model development and evaluation 
from sklearn.metrics import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#load data 
car_dataset = pd.read_csv(r'car_data.csv')

#explore data 

#view first five rows of the dataset 
car_dataset.head()

#view number of rows and columns 
car_dataset.shape
#there are 301 rows (301 data points) and 9 columns 

#view the statistical measures 
car_dataset.describe()

#get insight into type of columns and other information 
car_dataset.info()

#view missing values 
car_dataset.isnull().sum()
#there are no missing values in the dataset

#check distribution of categorical data
print (car_dataset.Fuel_Type.value_counts())
#there are 239 petrol-fueled cars, 60 diesel-fueled cars and 2 CNG-fueled cars

print (car_dataset.Seller_Type.value_counts())
#there are 195 cars sold by dealers and 106 cars sold by individuals 

print (car_dataset.Transmission.value_counts())
#the dataset shows 261 manual cars and 40 automatic cars

#converting text to numerical - this makes it easier to feed into the model 
#encoding categorical data
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

#separating data and label
X = car_dataset(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

#train model 
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

#evaluate model 

#LINEAR REGRESSION
#training data 
training_data_prediction = lin_reg_model.predict(X_train)

#r squared error 
error_score = metrics.r2_score(Y_train, training_data_prediction)
print ("r squared error", error_score)
#the r squared error is 0.87

#visualize actual price and predicted price 
plt.scatter(Y_train, training_data_prediction)
plt.xlabel('actual price')
plt.ylabel("predicted price")
plt.title("actual price vs predicted price")
plt.show()

#test data
test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print ("r squared error", error_score)

#visualize the results 
plt.scatter(Y_test, test_data_prediction)
plt.xlabel('actual price')
plt.ylabel("predicted price")
plt.title("actual price vs predicted price")
plt.show()
#the scatter points are very close together and it can be see from the plot that the predicted price is very close to the 
#actual price of the car. Therefore, the model performance is good 

#LASSO REGRESSION 
#training data
lass_reg_model = Lasso()
lass_reg_model.fit(X_train,Y_train)
training_data_prediction = lass_reg_model.predict(X_train)
#model evaluation
error_score = metrics.r2_score(Y_train, training_data_prediction)
print ("r squared error", error_score)
#the r squared error is 0.84

#visualize 
plt.scatter(Y_train, training_data_prediction)
plt.xlabel('actual price')
plt.ylabel("predicted price")
plt.title("actual price vs predicted price")
plt.show()
#in the plot, the scatter points are much closer as compared to linear regression model 
#therefore, lasso regression model fits the dataset better

#test data
test_data_prediction = lass_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print ("r squared error", error_score)
#r squared error is 0.87

#visualize 
plt.scatter(Y_test, test_data_prediction)
plt.xlabel('actual price')
plt.ylabel("predicted price")
plt.title("actual price vs predicted price")
plt.show()
#the scatter points lie in the same line. therefore it's a good fit