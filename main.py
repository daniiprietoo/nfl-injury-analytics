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
training_data = pd.read_csv('../data/processed/NFL_verse_training_data.csv')

y = training_data['Injury']
train_numeric = training_data[['week','qtr', 'down', 'ydstogo','yardline_100','spread_line',
                    'yards_gained', 'shotgun', 'no_huddle', 'qb_dropback',
                    'qb_scramble','season','overtime', 'div_game', 'wind', 'temp', 'score_differential']]
train_str = training_data[['game_half',
                    'play_type',
                    'pass_length','pass_location',
                    'run_location', 'run_gap','weekday','roof', 
                    'surface','stadium']]

# Encoding categorical features
le = sklearn.preprocessing.LabelEncoder()
for feat in train_str:
    train_str[feat] = le.fit_transform(train_str[feat].astype(str))

# Handle missing numeric data
for feat in train_numeric:
    train_numeric[feat].fillna(train_numeric[feat].mean(), inplace=True)

# Concatenate numeric and string data
pipeline = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.decomposition.PCA())
X_enc = pipeline.fit_transform(pd.concat([train_numeric, train_str], axis=1))

# Split into train/test sets
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
precision = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
recall = sklearn.metrics.recall_score(y_test, y_pred, average='macro')

# Compute mutual information
combined_data = pd.concat([train_numeric, train_str], axis=1)
ig = sklearn.feature_selection.mutual_info_classif(combined_data, y, random_state=42)

# Create a dictionary for features and their mutual information
ig_dict = {}
for i in range(len(combined_data.columns)):
    ig_dict[combined_data.columns[i]] = ig[i]

# Sort dictionary by mutual information
ig_dict_sorted = dict(sorted(ig_dict.items(), key=lambda item: item[1], reverse=True))

# Select the top n features with the highest information gain
n = 10
selected_features = list(ig_dict_sorted.keys())[:n]

# Create Streamlit app
st.title('NFL Injury Prediction Dashboard')

# Display processed data
st.subheader('Processed Data')
st.dataframe(pd.DataFrame(
    data=X_enc, 
    index=pd.concat([train_numeric, train_str], axis=1).index, 
    columns=pd.concat([train_numeric, train_str], axis=1).columns
).head())

# Model performance metrics
st.subheader('Model Performance Metrics')
st.write(f"R-Squared: {r_squared:.2f}")
st.write(f"Training Set Score: {train_score:.2f}")
st.write(f"Test Set Score: {test_score:.2f}")
st.write(f"Precision: {precision * 100:.2f}%")
st.write(f"Recall: {recall * 100:.2f}%")

# Display Mutual Information (Feature Importance)
st.subheader('Feature Importance (Mutual Information)')

# Display dictionary with feature names and their corresponding mutual information
st.write("Feature Importance based on Mutual Information:")
st.write(ig_dict_sorted)

# Display top 'n' selected features
st.write("Top 10 Features Based on Information Gain:")
st.write(selected_features)

# Allow users to download images or data if needed
if st.button("Download Feature Importance Data"):
    # Create a DataFrame for the feature importance
    feature_importance_df = pd.DataFrame(list(ig_dict_sorted.items()), columns=['Feature', 'Mutual Information'])
    feature_importance_df = feature_importance_df.sort_values(by='Mutual Information', ascending=False)
    
    # Save the dataframe as a CSV file
    csv = feature_importance_df.to_csv(index=False)
    
    # Save the CSV file locally
    with open('feature_importance.csv', 'w') as f:
        f.write(csv)
    
    # Provide the download link for the file
    st.download_button(
        label="Download Feature Importance CSV",
        data=csv,
        file_name='feature_importance.csv',
        mime='text/csv'
    )
