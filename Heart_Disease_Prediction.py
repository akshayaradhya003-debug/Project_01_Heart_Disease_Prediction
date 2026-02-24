import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

##Data collection and processing

heart_data = pd.read_csv(r"C:\Users\Admin\Desktop\akshay\heart_disease_data.csv")
print(heart_data.head())
print(heart_data.tail())
print(heart_data.shape)
print(heart_data.info())
print(heart_data.isnull().sum())

##statistical measures about the data
print(heart_data.describe())

##checking the target
print(heart_data['target'].value_counts())
## 1 defect
## 0 healthy

## spliting features and Target
x = heart_data.drop(columns='target',axis=1)
y = heart_data['target']

print(x)
print(y)
## splitting the data into train and test


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, X_train.shape, X_test.shape)

## Model Training

## Logistic Regeression

model = LogisticRegression()


## Training Logistic regression with training data

print(model.fit(X_train,Y_train))

## Model Evaluation

## Accuracy score

## accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training :", training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy on test data :", test_data_accuracy)

## building  predictive system 

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

input_data_as_numpy_array =np.asarray(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)
if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')