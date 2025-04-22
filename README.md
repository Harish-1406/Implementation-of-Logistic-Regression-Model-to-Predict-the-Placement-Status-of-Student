# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HARISH P K
RegisterNumber:  212224040104
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
dataset=dataset.drop('salary',axis=1)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(max_iter=10000) #By default it will have only 1000 iterations.
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)

Y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_pred,Y_test)
acc
clf.predict([[0,87,0,95,0,2,78,2,0,0,1,0]])

```
## Output:

![image](https://github.com/user-attachments/assets/5cb7e2b2-9bad-4a70-964a-2ee5752424ff)
![image](https://github.com/user-attachments/assets/da7867d8-ce91-4eb9-b568-ff90318b6fa4)
![image](https://github.com/user-attachments/assets/9851ccfb-5557-42f1-bb7f-f35434f3ec95)
![image](https://github.com/user-attachments/assets/55ed9cb4-5833-451b-b775-0d1371d2147b)
![image](https://github.com/user-attachments/assets/0d13a9ae-0196-46ba-b50c-625b9b21e28c)
![image](https://github.com/user-attachments/assets/d4ca77bc-d471-4505-9102-1e95ece9123b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
