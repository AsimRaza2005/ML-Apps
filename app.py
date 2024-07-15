import streamlit as st
import numpy as np
import pickle
    
# load_model = pickle.load(open('C:\\JN\\New folder\\Picke_model.pkl', "rb"))

st.write("""
         
#  My First ML App 
""")

st.title("Heart Failure Clinical Records")

def load_model():
    model_path = 'Heart_model.pkl'
    model = pickle.load(open(model_path, "rb"))
    return model

with st.sidebar:
    st.title("Sidebar")
    st.header("Heart failure is a chronic condition where the heart can't pump blood efficiently, leading to serious health complications and increased risk of death.")
    check = st.checkbox("Load Model")

if check:
    load_model()
    st.warning("Model Loaded Successfully")
    

    age = st.number_input("**Enter your age:**", min_value=30, max_value=85)
    anaemia = st.number_input("**you have  anaemia**", min_value=0, max_value=1)
    creatinine_phosphokinase = st.number_input("**Enter Creatine phosphokinase**")
    diabetes =  st.number_input("**You Have diabetes", min_value=0, max_value=1)
    ejection_fraction = st.number_input("**Enter Heart pumping Ratio**", min_value=20, max_value=90)
    high_blood_pressure = st.number_input("**You have Blood Pressurse**", min_value=0, max_value=1)
    platelets = st.number_input("**Enter Platelets Number**", min_value=25100, max_value=850000)
    serum_creatinine = st.number_input("**Enter Serum Creatinine**",min_value=0.5, max_value=9.4)
    serum_sodium = st.number_input("**Enter Serum Sodium**", min_value=113, max_value=148)
    sex = st.number_input("**Enter Gender(1-Male, 0-Female)**")
    smoking = st.number_input("**Are You Smooker(1,0)**")
    time = st.number_input("**What is the typical follow-up period after [treatment/procedure]?**")



    btn = st.button("Predict")

    if btn:
        data = [[age, anaemia, 	creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, 	sex, smoking, time	]]
        model = load_model()
        prediction = model.predict(data)
        if prediction == 0:
            st.write("You are still Survived")
            st.warning("Please consult with your doctor")
            st.error("Please take care of your health")
            # show the dr name and his contact detials
            st.info("Dr. Sheahid Afzal")
            st.info("📞Contact: 123-456-789")

        else:
            st.write("If you don't take care of your health properly, your heart could fail")

            st.write("Requirements for Patients with Heart Failure"

