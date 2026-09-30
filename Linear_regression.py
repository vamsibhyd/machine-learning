import pandas as pd 

New_dataset= pd.read_csv('New_dataset.csv')
New_dataset.columns
print(New_dataset.head())
New_dataset.info()


# Fill missing values

New_dataset['Arrival Delay in Minutes'] = New_dataset[

    'Arrival Delay in Minutes'

].fillna(

    New_dataset['Arrival Delay in Minutes'].median()

)

# giving the model all the other variables so that it can predict the required data
X = New_dataset[
    [
        'Age',
        'Flight Distance',
        'Inflight wifi service',
        'Departure/Arrival time convenient',
        'Ease of Online booking',
        'Gate location',
        'Food and drink',
        'Online boarding',
        'Seat comfort',
        'Inflight entertainment',
        'On-board service',
        'Leg room service',
        'Baggage handling',
        'Checkin service',
        'Cleanliness',
        'Departure Delay in Minutes'
    ]
]

# Target (what we want to predict)
y = New_dataset['Arrival Delay in Minutes']

print("X shape:", X.shape)
print("y shape:", y.shape)

# spliting the data 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

print("Training X:", X_train.shape)
print("Testing X:", X_test.shape)
print("Training y:", y_train.shape)
print("Testing y:", y_test.shape)


# creating the model 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# training the model

model.fit(X_train, y_train)

# makeing the predictions

y_pred = model.predict(X_test)

print(y_pred[:10]) # very important instead of showing all the datas predection we are only showing the 10 values of the predicted data 

#Evaluating the model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)


r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R² Score:", r2)# this will be the accuracy


# just creating a graph
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.3)

plt.xlabel("Actual Arrival Delay (minutes)")
plt.ylabel("Predicted Arrival Delay (minutes)")
plt.title("Actual vs Predicted Arrival Delay")

plt.show()



#clean code(to use just select all and command /)

# Import pandas for working with the dataset
# import pandas as pd

# # Load the dataset
# New_dataset = pd.read_csv('New_dataset.csv')


# # Fill missing values in Arrival Delay using the median
# New_dataset['Arrival Delay in Minutes'] = New_dataset[
#     'Arrival Delay in Minutes'
# ].fillna(
#     New_dataset['Arrival Delay in Minutes'].median()
# )


# # X = input features given to the model
# X = New_dataset[
#     [
#         'Age',
#         'Flight Distance',
#         'Inflight wifi service',
#         'Departure/Arrival time convenient',
#         'Ease of Online booking',
#         'Gate location',
#         'Food and drink',
#         'Online boarding',
#         'Seat comfort',
#         'Inflight entertainment',
#         'On-board service',
#         'Leg room service',
#         'Baggage handling',
#         'Checkin service',
#         'Cleanliness',
#         'Departure Delay in Minutes'
#     ]
# ]


# # y = target variable that we want the model to predict
# y = New_dataset['Arrival Delay in Minutes']


# # Split the data into training and testing sets
# # 80% of the data is used for training
# # 20% is used for testing
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )


# # Create the Linear Regression model
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()


# # Train the model using the training data
# model.fit(X_train, y_train)


# # Use the trained model to predict arrival delays
# y_pred = model.predict(X_test)


# # Evaluate the model
# from sklearn.metrics import mean_absolute_error, r2_score

# mae = mean_absolute_error(y_test, y_pred)

# r2 = r2_score(y_test, y_pred)

# print("Mean Absolute Error:", mae)
# print("R² Score:", r2)


# # Visualize actual values vs predicted values
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 6))

# plt.scatter(y_test, y_pred, alpha=0.3)

# plt.xlabel("Actual Arrival Delay (minutes)")
# plt.ylabel("Predicted Arrival Delay (minutes)")
# plt.title("Actual vs Predicted Arrival Delay")

# plt.show()
