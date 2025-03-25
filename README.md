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
Placement Data:

![image](https://github.com/user-attachments/assets/aced6675-2860-4dbe-8e2d-9124cf8e3de6)

Checking the null() function:

![image](https://github.com/user-attachments/assets/e12e11ca-4c97-4d3f-9eb3-2ed7591c4bb3)

Y_prediction array:

![image](https://github.com/user-attachments/assets/155b3c05-55e7-4da5-a4df-f891ec808158)

Accuracy value:

![image](https://github.com/user-attachments/assets/7511f950-d99c-4bb7-9611-a696e5168e16)

Classification Report:

![image](https://github.com/user-attachments/assets/1702c056-e177-4555-83cc-f1944848c0d2)

Prediction of LR:

![image](https://github.com/user-attachments/assets/2b827c4a-94ff-42db-a353-df8cd38f1fe7)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
