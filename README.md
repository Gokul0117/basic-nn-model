# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
 Name:Gokul J
 Register Number: 212222230038
```
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd  

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Deeplearning').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input': 'int', 'Output': 'int'})
dataset1.head()

x = dataset1[['Input']].values
y = dataset1[['Output']].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=33)

Scaler = MinMaxScaler()
Scaler.fit(x_train)

x_train1 = Scaler.transform(x_train)

ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs = 1000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test1 = Scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
x_n1=[[4]]
x_n1_1 = Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
## OUTPUT:

## Dataset Information

![image](https://github.com/user-attachments/assets/37fd6aed-5a54-4ef6-b45b-33f6fd672b1e)

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/1f66e434-4d30-46fe-818c-ec740c49f068)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/1768f0b0-8550-40fb-acd4-482d31d75ea2)

![image](https://github.com/user-attachments/assets/250ea715-2c15-4f21-a9b1-1c5a58af16f6)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a03d92b5-489d-40fe-83ae-894689c83bf5)


## RESULT:

Thus a Neural Network regression model for the given dataset is written and executed successfully

