# Developing a Neural Network Regression Model

### AIM

To develop a neural network regression model for the given dataset.

### THEORY

Explain the problem statement

### Neural Network Model

Include the neural network model diagram.

### DESIGN STEPS

- STEP 1:Loading the dataset
  
- STEP 2:Split the dataset into training and testing
  
- STEP 3:Create MinMaxScalar objects ,fit the model and transform the data.
  
- STEP 4:Build the Neural Network Model and compile the model.
  
- STEP 5:Train the model with the training data.
  
- STEP 6:Plot the performance plot
  
- STEP 7:Evaluate the model with the testing data.

### PROGRAM

#### Name: SASIDEVI V
#### Register Number: 212222230136

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet=gc.open("ex1DL").sheet1
df=worksheet.get_all_values()
print(df)
ds1=pd.DataFrame(df[1:],columns=df[0])
ds1=ds1.astype({'input':'float'})
ds1=ds1.astype({'output':'float'})
ds1.head()
x = ds1[['input']].values
y = ds1[['output']].values
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
xtrain=scaler.transform(x_train)
model=Sequential([Dense(8,activation="relu",input_shape=[1]),Dense(10,activation="relu"),Dense(1)])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain,y_train,epochs=2000)
cf=pd.DataFrame(model.history.history)
cf.plot()
xtrain=scaler.transform(x_test)
model.evaluate(xtrain,y_test)
n=[[17]]
n=scaler.transform(n)
model.predict(n)
```
### Dataset Information

#### DATASET.HEAD():
![image](https://github.com/user-attachments/assets/a7d1d93c-943c-434d-ab0f-12ab5ff50f28)
#### DATASET.INFO()
![image](https://github.com/user-attachments/assets/9a11303d-19df-47d9-9ebc-4eb59b14ea47)
#### DATASET.DESCRIBE()
![image](https://github.com/user-attachments/assets/38f7852f-bbc7-4ebd-b1a9-d398f7eb7f39)

### OUTPUT

#### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/eaa64275-cfdc-4a30-9ee6-54029d5903a2)

#### Test Data Root Mean Squared Error 
![image](https://github.com/user-attachments/assets/c790197f-77f1-44c9-a60a-b1435f961e6f)

### RESULT
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
