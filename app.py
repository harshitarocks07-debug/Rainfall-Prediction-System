import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

data = pd.read_csv("Rainfall.csv")

data.columns = data.columns.str.strip()

data["winddirection"] = data["winddirection"].fillna(
    data["winddirection"].mode()[0]
)

data["windspeed"] = data["windspeed"].fillna(
    data["windspeed"].median()
)

data["rainfall"] = data["rainfall"].map({
    "yes": 1,
    "no": 0
})

data = data.dropna(subset=["rainfall"])

df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

data_balanced = pd.concat([df_majority, df_minority_upsampled])

data_balanced = data_balanced.sample(
    frac=1,
    random_state=42
)

data_balanced = data_balanced.drop(
    columns=["maxtemp", "temparature", "mintemp"]
)

X = data_balanced.drop("rainfall", axis=1)

y = data_balanced["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

st.title("Rainfall Prediction System")

st.write("Enter weather details below:")

day = st.number_input("Day")
pressure = st.number_input("Pressure")
dewpoint = st.number_input("Dew Point")
humidity = st.number_input("Humidity")
cloud = st.number_input("Cloud")
sunshine = st.number_input("Sunshine")
winddirection = st.number_input("Wind Direction")
windspeed = st.number_input("Wind Speed")

if st.button("Predict Rainfall"):

    features = pd.DataFrame([{
        "day": day,
        "pressure": pressure,
        "dewpoint": dewpoint,
        "humidity": humidity,
        "cloud": cloud,
        "sunshine": sunshine,
        "winddirection": winddirection,
        "windspeed": windspeed
    }])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Rainfall Expected")
    else:
        st.error("No Rainfall Expected")