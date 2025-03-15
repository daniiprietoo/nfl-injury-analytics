import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv('data/raw/game_injury_player_2019_2020_complete_update_vegas (1).csv')

# Select features and target
features = ['week', 'surface', 'stadium', 'Avg_Temp', 'Avg_Feels_Like', 'Avg_Wind_MPH', 'Avg_Humidity_Percent']
target = 'num_injuries'

# Drop rows with missing values in the target column
df = df.dropna(subset=[target])

# Separate features and target
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for categorical and numerical data
categorical_features = ['surface', 'stadium']
numerical_features = ['week', 'Avg_Temp', 'Avg_Feels_Like', 'Avg_Wind_MPH', 'Avg_Humidity_Percent']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline with preprocessing and linear regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Example prediction
example = pd.DataFrame({
    'week': [1],
    'surface': ['Turf'],
    'stadium': ['U.S. Bank Stadium'],
    'Avg_Temp': [56],
    'Avg_Feels_Like': [56],
    'Avg_Wind_MPH': [3.5],
    'Avg_Humidity_Percent': [91.25]
})

predicted_injuries = model.predict(example)
print(f'Predicted number of injuries: {predicted_injuries[0]}')