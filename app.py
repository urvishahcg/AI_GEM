import streamlit as st
# import pandas as pd 
# from matplotlib import pyplot as plt
# from plotly import graph_objs as go
# from sklearn.linear_model import LinearRegression
# import numpy as np 
import webbrowser


# data = pd.read_csv("data//Salary_Data.csv")
# x = np.array(data['YearsExperience']).reshape(-1,1)
# lr = LinearRegression()
# lr.fit(x,np.array(data['Salary']))


st.title("Salary Predictor")
# st.image("data//sal.jpg",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
    #     st.table(data)
        print("****")
    # graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

    # val = st.slider("Filter data using years",0,20)
    # data = data.loc[data["YearsExperience"]>= val]
    # if graph == "Non-Interactive":
    #     plt.figure(figsize = (10,5))
    #     plt.scatter(data["YearsExperience"],data["Salary"])
    #     plt.ylim(0)
    #     plt.xlabel("Years of Experience")
    #     plt.ylabel("Salary")
    #     plt.tight_layout()
    #     st.pyplot()
    # if graph == "Interactive":
    #     layout =go.Layout(
    #         xaxis = dict(range=[0,16]),
    #         yaxis = dict(range =[0,210000])
    #     )
    #     fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode='markers'),layout = layout)
    #     st.plotly_chart(fig)
    
if nav == "Prediction":
    st.header("Know your Salary")
    # val = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
    # val = np.array(val).reshape(1,-1)
    # pred =lr.predict(val)[0]

    # if st.button("Predict"):
    #     st.success(f"Your predicted salary is {round(pred)}")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience",0.0,20.0)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)
    # if st.button("submit"):
    #     to_add = {"YearsExperience":[ex],"Salary":[sal]}
    #     to_add = pd.DataFrame(to_add)
    #     to_add.to_csv("data//Salary_Data.csv",mode='a',header = False,index= False)
    #     st.success("Submitted")

link = "https://teams.microsoft.com/l/entity/1c4340de-2a85-40e5-8eb0-4f295368978b/Home?context=%7B%22subEntityId%22%3A%22https%253A%252F%252Fapp.powerbi.com%252Flinks%252F7v1_8Aq6iX%253Fctid%253D76a2ae5a-9f00-4f6b-95ed-5d33d77c4d61%2526pbi_source%253DlinkShare%2526bookmarkGuid%253D6c60fc61-4894-4759-ac57-4c8826347cee%22%7D"
st.markdown(link,unsafe_allow_html=True)

if st.button("PowerBi Report"):
    link = "https://teams.microsoft.com/l/entity/1c4340de-2a85-40e5-8eb0-4f295368978b/Home?context=%7B%22subEntityId%22%3A%22https%253A%252F%252Fapp.powerbi.com%252Flinks%252F7v1_8Aq6iX%253Fctid%253D76a2ae5a-9f00-4f6b-95ed-5d33d77c4d61%2526pbi_source%253DlinkShare%2526bookmarkGuid%253D6c60fc61-4894-4759-ac57-4c8826347cee%22%7D"

    webbrowser.open_new_tab(link)
    # st.markdown(link,unsafe_allow_html=True)
