✈️ Airline Arrival Delay Prediction using Linear Regression

This project uses Linear Regression to predict the arrival delay time of an airline flight in minutes based on different flight and passenger-service features.

The project demonstrates a basic Machine Learning regression workflow using Python and Scikit-learn.

📌 Project Objective

The goal of this project is to predict:

How many minutes a flight will be delayed upon arrival?

The model uses information such as:

* Passenger age
* Flight distance
* Departure delay
* Online booking experience
* Inflight services
* Seat comfort
* Food and drink
* Baggage handling
* Cleanliness
* Check-in service
* And other airline service features

⸻

🧠 Machine Learning Algorithm

Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

In this project:

Input features (X)
        ↓
Linear Regression Model
        ↓
Predicted Arrival Delay

For example:

Flight information
       ↓
Model
       ↓
Predicted arrival delay = 25.4 minutes

⸻

📊 Dataset

The project uses a CSV dataset named:

New_dataset.csv

The target variable is:

Arrival Delay in Minutes

Input Features

The model uses the following features:

Age
Flight Distance
Inflight wifi service
Departure/Arrival time convenient
Ease of Online booking
Gate location
Food and drink
Online boarding
Seat comfort
Inflight entertainment
On-board service
Leg room service
Baggage handling
Checkin service
Cleanliness
Departure Delay in Minutes

Target Variable

Arrival Delay in Minutes

The target is the value that the model learns to predict.

⸻

🔧 Data Preprocessing

Before training the model, missing values in the Arrival Delay in Minutes column are handled.

The missing values are replaced using the median of the column:

New_dataset['Arrival Delay in Minutes'] = New_dataset[
    'Arrival Delay in Minutes'
].fillna(
    New_dataset['Arrival Delay in Minutes'].median()
)

Using the median helps provide a reasonable value for missing observations without being strongly affected by extreme delay values.

⸻

🧪 Train-Test Split

The dataset is divided into two parts:

* 80% → Training data
* 20% → Testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

Why split the data?

The model learns from the training data and is then tested on data that it has not seen before.

This helps us evaluate how well the model performs on unseen data.

random_state

random_state=1 makes the split reproducible.

If you run the code again with the same dataset and the same random_state, you will get the same train/test split.

⸻

🤖 Model Training

The Linear Regression model is created using Scikit-learn:

from sklearn.linear_model import LinearRegression
model = LinearRegression()

The model is trained using:

model.fit(X_train, y_train)

During training, the model learns the relationship between the input features and arrival delay.

⸻

🔮 Making Predictions

After training, the model predicts arrival delays for the test data:

y_pred = model.predict(X_test)

Only the first 10 predictions are displayed:

print(y_pred[:10])

This prevents the program from printing the entire prediction dataset.

⸻

📈 Model Evaluation

Two evaluation metrics are used.

1. Mean Absolute Error (MAE)

mae = mean_absolute_error(y_test, y_pred)

MAE tells us the average absolute difference between the actual arrival delay and the predicted arrival delay.

For example:

MAE = 12.5

means that, on average, the predictions are approximately 12.5 minutes away from the actual values.

Lower MAE generally means smaller prediction errors.

⸻

2. R² Score

r2 = r2_score(y_test, y_pred)

R² measures how well the model explains the variation in the target variable.

The value can be interpreted roughly as:

R² = 1
→ Perfect fit
R² close to 1
→ Model explains a large amount of the variation
R² close to 0
→ Model explains little of the variation
R² < 0
→ Model performs poorly compared with a simple baseline

Note: R² is not the same as classification accuracy. Since this is a regression problem, it is better to describe R² as a goodness-of-fit metric rather than “accuracy.”

⸻

📊 Visualization

The project also creates a scatter plot comparing:

Actual Arrival Delay
        vs
Predicted Arrival Delay

The graph is created using Matplotlib:

plt.scatter(y_test, y_pred, alpha=0.3)

This helps visually inspect how close the model’s predictions are to the actual values.

⸻

🛠️ Technologies Used

* Python
* Pandas – Data loading and preprocessing
* NumPy – Numerical operations
* Scikit-learn – Machine Learning model and evaluation
* Matplotlib – Data visualization

⸻

📁 Project Structure

Linear-Regression/
│
├── New_dataset.csv
├── linear_regression.py
└── README.md

Replace linear_regression.py with the actual filename of your Python file if it has a different name.

⸻

▶️ How to Run the Project

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Open the project

cd Linear-Regression

3. Install the required libraries

pip install pandas numpy scikit-learn matplotlib

4. Run the Python program

python linear_regression.py

⸻

🔄 Machine Learning Workflow

The complete workflow of this project is:

Load Dataset
     ↓
Explore Dataset
     ↓
Handle Missing Values
     ↓
Select Features (X)
     ↓
Select Target (y)
     ↓
Split Training & Testing Data
     ↓
Create Linear Regression Model
     ↓
Train Model
     ↓
Make Predictions
     ↓
Calculate MAE & R²
     ↓
Visualize Predictions

⸻

🚀 Future Improvements

Possible improvements to this project include:

* Feature scaling
* Feature selection
* Outlier detection
* Comparing Linear Regression with other regression algorithms
* Cross-validation
* Hyperparameter tuning
* Adding more evaluation metrics
* Deploying the model as a web application

⸻

📚 What I Learned

Through this project, I learned the basic workflow of building a supervised machine learning regression model:

* Loading and exploring a dataset
* Handling missing values
* Selecting features and target variables
* Splitting data into training and testing sets
* Training a Linear Regression model
* Making predictions
* Evaluating model performance
* Visualizing actual vs predicted values

⸻

👨‍💻 Author

Vamsi Bolla

GitHub: @vamsibhyd
