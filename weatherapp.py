import streamlit as st
import numpy as np
import pickle


st.title("Weather Classification Report")

def load_model():
    
    model = pickle.load(open("Weather_model.pkl", 'rb'))
    return model

check = st.checkbox("Load Model")

if check:
    load_model()
    st.warning("Model Loaded Successfully")
    
    
    Temperature = st.number_input("**Enter Temperature of Environment**")   
    Humidity = st.number_input("**Enter Humadity in Environment**")
    WindSpeed = st.number_input("**Enter Wind Speed**")
    Precipitation = st.number_input("**What percentage of the time do you experience precipitation**")
    CloudCover = st.number_input("**Enter Cloud Cover(partlycloudy, overcast, Clear, 0,1,2)**")
    AtmosphericPressure = st.number_input("**What is the typical atmospheric pressure in your area**")
    UVIndex = st.number_input("**What is the average UV index in your location**", min_value=0, max_value=10)
    Season = st.number_input("**Enter Season( Autumn,spring, summer, winter, 0,1,2,3)**")
    Visibility = st.number_input("**What is the average visibility in kilometers in your area**")
    Location = st.number_input("**Enter Location (inland, Mountain, Coastal, 0,1,2)**")


    btn = st.button("Predict")

    if btn:
        data = [[Temperature, Humidity, WindSpeed, Precipitation, CloudCover, AtmosphericPressure, UVIndex, Season, Visibility, Location]]
        model = load_model()
        prediction = model.predict(data)
        
        if prediction == 0:
            st.write("Cloudy conditions are frequent in our area.")
            
        elif prediction == 1:
            st.write("Rainy conditions are frequent in our area.")
            
        elif prediction == 2 :
            st.write("Snowy conditions are frequent in our area.") 
            
        elif prediction == 3:
            st.write("Sunny conditions are frequent in our area.")       
                
