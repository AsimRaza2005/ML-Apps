import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Used Car Sales Analysis")

def load_model():
    
    model = pickle.load(open("CarSelling_model.pkl",'rb'))
    return model

check = st.checkbox("Load Model")

if check:
    st.warning("Model Loaded Successfully")
    model = load_model()

    
    Year = st.number_input("**Enter Car Model Year**")    
    Present_Price = st.number_input("**Enter Car Present Price**")
    Kms_Driven = st.number_input("**How many Kilometeres Car is Driven**")
    Fuel_Type = st.number_input("**Enter Fuel Type(CNG, Diesel, Petrol, 0,1,2,)**")
    Seller_Type = st.number_input("**Enter Seller Type(Dealer,0)**")
    Transmission = st.number_input("**Car Transmission(AutoMatic, Mannual, 0,1)**")
    Owner = st.number_input("**Enter car Owner(0)**")
    
    
    if st.button('Calculate Call Selling Price'):
        
        features = [[Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner]]
        Selling_price = model.predict(features)[0]
        st.write(f'Used Car Sales Price : {Selling_price:.2f} Lacs')

    
    
    