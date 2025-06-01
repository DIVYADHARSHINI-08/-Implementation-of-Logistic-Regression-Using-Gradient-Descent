# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

6.Define a function to predict the Regression value. 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DIVYA DHARSHINI R 
RegisterNumber: 212223040042
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)
print(y_pred)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
## Output:
## Read the file and display
![image](https://github.com/user-attachments/assets/9bdb2af2-42aa-48c7-be9a-fe1df4202a81)
## Categorizing columns
![image](https://github.com/user-attachments/assets/a119964e-f688-4c76-b682-f12aaa3b7fba)
## Labelling columns and displaying dataset
![image](https://github.com/user-attachments/assets/7d6422b2-c702-4ef5-b8b6-7ebd94a23acb)
## Display dependent variable
![image](https://github.com/user-attachments/assets/4d887ed7-6655-4505-b7df-aa9c22c69632)
## Accuracy
![image](https://github.com/user-attachments/assets/de1ff5bb-4250-4b28-882f-0def467a3412)
## Printing Y
![image](https://github.com/user-attachments/assets/f2a4b7f4-33fa-4e19-9b6d-45309b3d6811)
## Printing Y_prednew
![image](https://github.com/user-attachments/assets/9f122748-9413-4513-8a6c-2938a96d987c)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

