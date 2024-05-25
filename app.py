import streamlit as st
import pickle
import pandas as pd
import pandas
import sklearn
import numpy as np

lr1 = pickle.load(open('lr1_model_25May.pkl','rb'))
dt1 = pickle.load(open('dt1_model_25May.pkl','rb'))
rf1 = pickle.load(open('rf1_model_25May.pkl','rb'))


st.title('Insurance Charge Prediction App')


st.header('Fill the Details to generate the Predicted Insurance Charge')

##options = st.sidebar.selectbox(['Lin_Reg','Decision_Tree','Random_Forest'])
options = st.sidebar.selectbox('Choose a model', ['Lin_Reg', 'Decision_Tree', 'Random_Forest'])



# Form Widgets


age = st.slider('Age',18,64)
sex = st.selectbox('Sex',['Male','Female'])
bmi = st.slider('BMI',6,53)
children = st.selectbox('Children',[0,1,2,3,4,5])
smoker = st.selectbox('Smoker',['Yes','No'])
region = st.selectbox('Region',['NWest','SEast','SWest','NEast'])

if st.button('Predict'):
    if sex == 'Male':
        sex = 1
    else:
        sex = 0
    if smoker == "Yes":
        smoker = 1
    else:
        smoker = 0
    if region == "NWest":
       region = 1
    elif region == "NEast":
         region = 0
    elif region == "SEast":
        region == 2
    else:
        region = 3          


    test = np.array([age,sex,bmi,children,smoker,region])
    test = test.reshape(1,6)
    if options == "Lin_Reg":
        st.success(lr1.predict(test)[0])
    elif options == "Decision_Tree":
        st.success(dt1.predict(test)[0])
    else:
        st.success(rf1.predict(test)[0])    
