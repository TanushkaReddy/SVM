import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="SVM ML App", layout="centered")

st.title("SVM Machine Learning Application")
st.write("Upload a dataset and perform SVM Classification or Regression.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write("Shape:", df.shape)

    # Select Target Column
    target_column = st.selectbox("Select Target Column", df.columns)

    # Select Model Type
    model_type = st.radio(
        "Choose Problem Type",
        ["Classification", "Regression"]
    )

    # Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle Categorical Columns
    le = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = le.fit_transform(X[col])

    if y.dtype == "object":
        y = le.fit_transform(y)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Model Button
    if st.button("Train Model"):

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if model_type == "Classification":

            model = SVC(kernel='linear')

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            st.success("SVM Classification Model Trained Successfully")

            st.subheader("Accuracy")
            st.write(round(accuracy * 100, 2), "%")

        # ====================================================
        # REGRESSION
        # ====================================================

        else:

            model = SVR(kernel='rbf')

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            st.success("SVM Regression Model Trained Successfully")

            st.subheader("R2 Score")
            st.write(round(r2, 4))