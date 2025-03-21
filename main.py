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

# Load and preprocess data
training_data = pd.read_csv('data/processed/NFL_verse_training_data.csv')

y = training_data['Injury']

# Split into train/test sets
X_enc = training_data.drop(columns=['Injury'])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_enc, y, test_size=0.2, random_state=41)

# Train the model
linr = sklearn.linear_model.LinearRegression()
linr.fit(X_train, y_train)

# Predictions
y_pred = linr.predict(X_test)

# Model evaluation metrics
r_squared = linr.score(X_test, y_test)
train_score = linr.score(X_train, y_train)
test_score = linr.score(X_test, y_test)


# Create Streamlit app
st.title('NFL Injury Prediction Dashboard')

# Display processed data
st.subheader('Processed Data')
st.dataframe(training_data.head())

# Model performance metrics
st.subheader('Model Performance Metrics')
st.write(f"R-Squared: {r_squared:.2f}")
st.write(f"Training Set Score: {train_score:.2f}")
st.write(f"Test Set Score: {test_score:.2f}")

# Display Mutual Information (Feature Importance)
st.subheader('Feature Importance (Mutual Information)')

# Display dictionary with feature names and their corresponding mutual information
st.write("Feature Importance based on Mutual Information:")

# Display top 'n' selected features
st.write("Top 10 Features Based on Information Gain:")

# Allow users to download images or data if needed
st.button("Download Feature Importance Data")
    # # Create a DataFrame for the feature importance
    # feature_importance_df = pd.DataFrame(list(ig_dict_sorted.items()), columns=['Feature', 'Mutual Information'])
    # feature_importance_df = feature_importance_df.sort_values(by='Mutual Information', ascending=False)
    
    # # Save the dataframe as a CSV file
    # csv = feature_importance_df.to_csv(index=False)
    
    # # Save the CSV file locally
    # with open('feature_importance.csv', 'w') as f:
    #     f.write(csv)
    
    # # Provide the download link for the file
    # st.download_button(
    #     label="Download Feature Importance CSV",
    #     data=csv,
    #     file_name='feature_importance.csv',
    #     mime='text/csv'
    # 