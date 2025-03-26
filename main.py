import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import sklearn.calibration
import sklearn.metrics
import sklearn.feature_selection
import sklearn.preprocessing
import sklearn.decomposition

@st.cache_data
def load_data():
    clean_data = pd.read_csv('./data/processed/NFL_verse_training_data.csv')
    encoded_data = pd.read_csv('./data/processed/NFL_verse_training_data_encoded.csv') 
    final_data = pd.read_csv('./data/processed/NFL_verse_training_data_final.csv')
    y = pd.read_csv('./data/processed/y.csv')
    return clean_data, encoded_data, final_data, y

def train_model(data, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, y, test_size=0.2, random_state=42)
    linr = sklearn.linear_model.LinearRegression()
    
    # Fit the model
    linr.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = linr.predict(X_train)
    y_pred_test = linr.predict(X_test)
    
    # Calculate performance metrics
    train_score = sklearn.metrics.r2_score(y_train, y_pred_train)
    test_score = sklearn.metrics.r2_score(y_test, y_pred_test)
    
    return linr, train_score, test_score

# Load data
clean_data, encoded_data, final_data, y = load_data()
# Train the model
model, train_score, test_score = train_model(final_data, y)

# Create Streamlit app
st.title('NFL Injury Prediction Report')
st.write("This is the report of our NFL injury prediction model. We used historical data from the NFL-verse to train our model ranging from 2010 to 2023.")
st.write("The model predicts injuries based on various factors and data extracted from the datasets.")
st.write("The data used for training includes play-by-play data and schedule data.")
st.write("We ended up using only Pass and Run plays for our model.")
st.write("The model is trained using a Linear Regression algorithm.")

# Display processed data
st.subheader('Processed Data')
st.write("This is the processed data used for training the model. ")
st.dataframe(clean_data.head())

# Display the plots
st.subheader('Data Visualizations of Pass or Run Plays')
st.write("This section contains various visualizations that help understand the data better.")
st.image('./visualizations/injuries_total.png', caption='Total Injuries')
st.image('./visualizations/injuries_by_year.png', caption='Injuries by Position')
st.image('./visualizations/injuries_by_surface.png', caption='Injuries by Surface')
st.image('./visualizations/injuries_by_body_part.png', caption='Injuries by Body Part')

# Model training
st.subheader('Model Training')
st.write('Before training we first encoded the categortical variables and standardized the data.')
st.dataframe(encoded_data.head())
st.write("Then we performed an information gain analysis to see which features are most important for our model.")
st.image('./visualizations/information_gain.png', caption='Information Gain Analysis')
st.write("We then split the data into training and testing sets (80-20).")
st.write("We used the top 20 features selected by the information gain analysis:")
st.dataframe(final_data.columns)
# Model performance metrics
st.subheader('Model Performance Metrics')
st.write(f"Training Set Score: {(train_score * 100):.2f}")
st.write(f"Test Set Score: {(test_score * 100):.2f}")
st.write(f"R-Squared: {((train_score + test_score) / 2 * 100):.2f}")

# Model confusion matrix
st.subheader('Confusion Matrix')
st.image('./visualizations/confusion_matrix.png', caption='Confusion Matrix')
