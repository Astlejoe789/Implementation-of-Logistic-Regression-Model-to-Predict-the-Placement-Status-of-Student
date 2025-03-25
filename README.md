# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ASTLE JOE A S
RegisterNumber:  212224240019
*/
import pandas as pd
data=pd.read_csv(r"C:\Users\astle\Downloads\Placement_Data.csv") 
data.head() 
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum() 
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"]) 
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1] 
x
y=data1["status"] 
y
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)  
x_test
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print(classification_report1) 
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
![image](https://github.com/user-attachments/assets/351377eb-5fb6-423d-a511-2a443ed16b00)

![image](https://github.com/user-attachments/assets/f62d6907-61f4-4987-9633-8c10d8a7140b)

![image](https://github.com/user-attachments/assets/f845e995-dfd9-4442-801c-a44f584ba19f)

![image](https://github.com/user-attachments/assets/b7412b4a-f7ec-4e9b-9148-30d23c0448cb)

![image](https://github.com/user-attachments/assets/964e90d7-d5e2-4aee-91ad-ad27b69a3c98)

![image](https://github.com/user-attachments/assets/2d5b81d1-f66b-43f9-b530-a6c5c7833541)

![image](https://github.com/user-attachments/assets/d755dba2-1cfd-46ea-a2e9-f15f309326ce)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
