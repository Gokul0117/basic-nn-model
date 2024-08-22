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
 Name: Gokul J
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

![Screenshot 2024-08-22 203528](https://github.com/user-attachments/assets/c3da8602-874a-43d8-8492-ce623a12226f)

### Training Loss Vs Iteration Plot

![Screenshot 2024-08-22 203654](https://github.com/user-attachments/assets/bb7211f6-ead8-4bb6-9e1c-6d30d6ce1af3)

### Test Data Root Mean Squared Error

![Screenshot 2024-08-22 204119](https://github.com/user-attachments/assets/2fb3bc62-37c9-411c-8d07-8f90b55a6cb0)

![Screenshot 2024-08-22 204146](https://github.com/user-attachments/assets/b4af41a9-5bb4-49f1-a2a4-c8eabdf3ebf1)


### New Sample Data Prediction

![Screenshot 2024-08-22 204207](https://github.com/user-attachments/assets/262bd816-1267-4015-9870-767aaf75331e)


## RESULT:

Thus a Neural Network regression model for the given dataset is written and executed successfully


