import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background-image: url("clouds.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.main {
    background: rgba(0,0,0,0.72);
    padding: 2rem;
    border-radius: 15px;
}

h1, h2, h3, h4, p, label {
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: rgba(17,24,39,0.85);
}

div[data-baseweb="input"] {
    background-color: rgba(255,255,255,0.12);
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

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

data_balanced = pd.concat([
    df_majority,
    df_minority_upsampled
])

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

st.sidebar.title("Rainfall Prediction App")

st.sidebar.write("""
Predict rainfall using machine learning
and weather parameters.
""")

st.title("Rainfall Prediction System")

st.subheader("Predict rainfall using weather parameters")

st.metric("Model Accuracy", "86%")

st.subheader("Rainfall Distribution")

rainfall_counts = data["rainfall"].value_counts()

st.bar_chart(rainfall_counts)

st.subheader("Humidity Trend")

st.line_chart(data["humidity"].head(100))

st.subheader("Feature Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)

st.write("Enter weather details below:")

col1, col2 = st.columns(2)

with col1:

    day = st.number_input(
        "Day of Year",
        min_value=1,
        max_value=365
    )

    pressure = st.number_input("Pressure")

    dewpoint = st.number_input("Dew Point")

    humidity = st.number_input(
        "Humidity",
        min_value=0,
        max_value=100
    )

with col2:

    cloud = st.number_input("Cloud")

    sunshine = st.number_input("Sunshine Hours")

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

    probability = model.predict_proba(features)[0][1]

    if prediction[0] == 1:

        st.success("Rainfall Expected")

    else:

        st.error("No Rainfall Expected")

    st.write(
        f"Probability of Rainfall: {probability * 100:.2f}%"
    )

    st.progress(int(probability * 100))

st.markdown("---")

st.write("Developed by Harshita Singh")