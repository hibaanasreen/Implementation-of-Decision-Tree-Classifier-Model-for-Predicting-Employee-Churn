# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```python

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Hiba Nasreen M
RegisterNumber: 21224040117 

```
```python
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```python
print("data.info():")
data.info()
```
```python
print("isnull() and sum():")
data.isnull().sum()
```
```python
print("data value counts():")
data["left"].value_counts()
```
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```python
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```python
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```

```python
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```python
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```python
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:

<img width="1373" height="229" alt="image" src="https://github.com/user-attachments/assets/b4f81c24-2595-4ecc-a293-ea094b665a66" />

<img width="522" height="388" alt="image" src="https://github.com/user-attachments/assets/572360df-718c-4302-a13b-a2b97fcc8621" />


<img width="323" height="272" alt="image" src="https://github.com/user-attachments/assets/97d0b5e8-6ee4-4fdc-8e41-570d3f280ecf" />


<img width="323" height="272" alt="image" src="https://github.com/user-attachments/assets/491f3217-23be-4d70-8532-3c324a18fcfb" />


<img width="455" height="118" alt="image" src="https://github.com/user-attachments/assets/dc3304cb-761c-4ba4-8be4-a3fbcb4b2e7e" />


<img width="1350" height="239" alt="image" src="https://github.com/user-attachments/assets/7747786a-64ae-4dde-b5e4-d3c227d3209d" />


<img width="1196" height="232" alt="image" src="https://github.com/user-attachments/assets/768fef37-0b73-471a-ac40-ca5411cafe3d" />


<img width="290" height="68" alt="image" src="https://github.com/user-attachments/assets/9d0f6dc6-0ab8-49dd-877a-a0d1010f9dae" />

<img width="144" height="40" alt="image" src="https://github.com/user-attachments/assets/47aa8f22-10ef-44fd-8ea9-f92d121213c6" />



<img width="821" height="598" alt="image" src="https://github.com/user-attachments/assets/2ea4bf67-3a7b-4215-bf76-438677620d3e" />







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
