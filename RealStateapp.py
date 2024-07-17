import streamlit as st
import pickle
import pandas as pd

st.title("Real Estate Price Detector")

def load_model():
    model = pickle.load(open("RealState_model.pkl", 'rb'))
    return model

check = st.checkbox("Load Model")

if check:
    st.warning("Model Loaded Successfully")
    model = load_model()
    
    
    df = pd.read_csv("date.csv")
    
    format = df["fromat"].to_list()
    preprocess = df["preprocess"].to_list()
    
    transaction_format = st.selectbox("Select option", format)
    # Transaction_Date = st.number_input("Transaction Date")
    
    
    House_Age = st.number_input("Age of House in Years")
    MRT_station_at_Nearest_distance = st.number_input("Distance to Nearest MRT Station")
    Convenience_stores_number = st.number_input("Number of Convenience Stores Nearby")
    Latitude = st.number_input("Latitude of the Property")
    Longitude = st.number_input("Longitude of the Property")
    
    
    transaction_preprocess= preprocess[format.index(transaction_format)]
    
    # st.write(transaction_preprocess)
    
    if st.button('Calculate Price Per Square Foot'):
        
        features = [[transaction_preprocess, House_Age, MRT_station_at_Nearest_distance, Convenience_stores_number, Latitude, Longitude]]
        price_per_sqft = model.predict(features)[0]
        st.write(f'Price per Square Foot: {price_per_sqft:.2f}')
