import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create dropdown menus for gender
gender_options = [1, 2]
gender_labels = ['Female', 'Male']
gender_dict = dict(zip(gender_options, gender_labels))

# Create dropdown menus for rank
rank_options = [0, 1]
rank_labels = ['Not Pass', 'Pass']
rank_dict = dict(zip(rank_options, rank_labels))

# User inputs
st.image("image.png")
st.write("## Rank Prediction")
gender = st.selectbox("Gender", options=gender_options, format_func=lambda x: gender_dict[x])
age = st.number_input("Age", min_value=0)
wyr = st.number_input("Wyr", min_value=0.0)
mstat = st.number_input("Mstat", min_value=0)
consist = st.number_input("Consist", min_value=0)
achieve = st.number_input("Achieve", min_value=0)
defer = st.number_input("Defer", min_value=0)
order = st.number_input("Order", min_value=0)
exhib = st.number_input("Exhib", min_value=0)
auto = st.number_input("Auto", min_value=0)
affi = st.number_input("Affi", min_value=0)
inter = st.number_input("Inter", min_value=0)
suco = st.number_input("Suco", min_value=0)
domi = st.number_input("Domi", min_value=0)
abas = st.number_input("Abas", min_value=0)
nurt = st.number_input("Nurt", min_value=0)
chang = st.number_input("Chang", min_value=0)
endu = st.number_input("Endu", min_value=0)
heter = st.number_input("Heter", min_value=0)
agg = st.number_input("Agg", min_value=0)

# Load the data
data = pd.read_csv("data.csv")

# Remove the stdate column
data = data.drop("stdate", axis=1)

# Prepare the data for training
X = data.drop("rank", axis=1)
y = data["rank"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on user inputs
user_input = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "wyr": [wyr],
    "mstat": [mstat],
    "consist": [consist],
    "achieve": [achieve],
    "defer": [defer],
    "order": [order],
    "exhib": [exhib],
    "auto": [auto],
    "affi": [affi],
    "inter": [inter],
    "suco": [suco],
    "domi": [domi],
    "abas": [abas],
    "nurt": [nurt],
    "chang": [chang],
    "endu": [endu],
    "heter": [heter],
    "agg": [agg]
})

prediction = model.predict(user_input)

# Display the prediction
st.write("### Prediction:")
st.write(rank_dict[prediction[0]])

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy
st.write("### Accuracy:")
st.write(accuracy)