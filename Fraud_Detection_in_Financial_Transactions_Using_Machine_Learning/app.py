# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import streamlit_authenticator as stauth

# ---- Authentication ----
# Define users with usernames, hashed passwords, and names
usernames = ["user1", "user2"]
names = ["John Doe", "Jane Smith"]
passwords = ["password1", "password2"]

# Hash the passwords
hashed_passwords = stauth.Hasher(passwords).generate()

# Create the authenticator
authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "fraud_detection_app", "abcdef", cookie_expiry_days=1)

# ---- Streamlit App ----
# Page title
st.title("Fraud Detection in Financial Transactions")

# Login Page
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"Welcome, {name}!")

    # Step 1: Upload Dataset
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("C:\Users\PC\OneDrive - Horizon Campus\Desktop\Fraud_Detection_in_Financial_Transactions_Using_Machine_Learning\creditcard.csv\creditcard.csv", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        # Step 2: Data Preprocessing
        st.subheader("Data Preprocessing")
        if st.checkbox("Handle Missing Values"):
            data = data.dropna()  # Drop rows with missing values
            st.write("Missing values removed.")

        # Display class distribution
        st.write("Class Distribution:")
        st.write(data['Class'].value_counts())
        
        # Step 3: Separate Features and Labels
        X = data.drop("Class", axis=1)
        y = data["Class"]

        # Handle Class Imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Feature Scaling
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

        # Step 4: Train the Model
        st.subheader("Train the Model")
        if st.button("Train Random Forest Classifier"):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            st.write("Model trained successfully!")

            # Step 5: Make Predictions and Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Display Evaluation Metrics
            st.subheader("Evaluation Metrics")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
    else:
        st.info("Awaiting CSV file to be uploaded.")

elif authentication_status == False:
    st.error("Username or password is incorrect")

elif authentication_status == None:
    st.warning("Please enter your username and password")
