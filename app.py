#from asyncio.windows_events import NULL
from queue import Empty
#from tkinter.ttk import Style
import streamlit as st
import webbrowser
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
#from PIL import Image
import itertools
# warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib

st.title("Consumer insights analysis using wearable devices data in B2B")
nav = st.sidebar.radio("Navigation",["Home","PowerBi Visualization","Visualization","Clustering","Sales Prediction","Team"])


if nav == "Home":
    # image = Image.open('C:\Users\urshah\Downloads\AI_GEM_9\images\images.jpg')
    # st.image(image, caption='',width=500)
    st.write("")


if nav == "PowerBi Visualization":
    st.markdown("<h2>PowerBi Visualization</h2>", unsafe_allow_html=True)

    st.write("1. Visualization report of Based on region")
    link1 = "https://teams.microsoft.com/l/entity/1c4340de-2a85-40e5-8eb0-4f295368978b/Home?context=%7B%22subEntityId%22%3A%22https%253A%252F%252Fapp.powerbi.com%252Fgroups%252Fme%252Freports%252F07a404b2-1fa6-42e0-993a-bfcf5e0f4cf3%252FReportSection%253Faction%253DOpenReport%2526pbi_source%253DMSTeams%2526bookmarkGuid%253D9b677413-b7a9-4b5e-9555-3f6c9425695c%22%7D"
    st.markdown(link1,unsafe_allow_html=True)

    st.write("2. Visualization report of Raw Consumer data to identify the Trends.")    
    link2 = "https://teams.microsoft.com/l/entity/1c4340de-2a85-40e5-8eb0-4f295368978b/Home?context=%7B%22subEntityId%22%3A%22https%253A%252F%252Fapp.powerbi.com%252Fgroups%252Fme%252Freports%252F4e005cc9-92eb-4e32-9c02-686a2a02d39d%252FReportSection%253Faction%253DOpenReport%2526pbi_source%253DMSTeams%22%7D"
    st.markdown(link2,unsafe_allow_html=True)
    
    st.write("3. Visualization report of Raw Consumer data to identify the Trends.") 
    link3 = "https://teams.microsoft.com/l/entity/1c4340de-2a85-40e5-8eb0-4f295368978b/Home?context=%7B%22subEntityId%22%3A%22https%253A%252F%252Fapp.powerbi.com%252Fgroups%252Fme%252Freports%252Fa4e7a0bc-cfd8-4f6d-8239-b24b7c24ecdf%252FReportSection%253Faction%253DOpenReport%2526pbi_source%253DMSTeams%2526bookmarkGuid%253D9f2dc4df-73fc-4dde-bc96-51dffd687e78%22%7D"   
    st.markdown(link3,unsafe_allow_html=True)

    st.write("4. Consumer Vitals") 
    link4 ="https://teams.microsoft.com/l/entity/1c4340de-2a85-40e5-8eb0-4f295368978b/Home?context=%7B%22subEntityId%22%3A%22https%253A%252F%252Fapp.powerbi.com%252Fgroups%252Fme%252Freports%252F8ffc34df-6d11-488e-8704-53ceba23c365%252FReportSection%253Faction%253DOpenReport%2526pbi_source%253DMSTeams%22%7D"
    st.markdown(link4,unsafe_allow_html=True) 

    st.write("5. Activity report")
    link5 = "https://teams.microsoft.com/dl/launcher/launcher.html?url=%2F_%23%2Fl%2Fentity%2F1c4340de-2a85-40e5-8eb0-4f295368978b%2FHome%3Fcontext%3D%257B%2522subEntityId%2522%253A%2522https%25253A%25252F%25252Fapp.powerbi.com%25252Fgroups%25252Fme%25252Freports%25252F84456807-b8e7-4830-a37c-2cd49d514181%25252FReportSection%25253Faction%25253DOpenReport%252526pbi_source%25253DMSTeams%2522%257D&type=entity&deeplinkId=ad589a02-7a82-4e01-9e41-f856839580e7&directDl=true&msLaunch=true&enableMobilePage=true&suppressPrompt=true"
    st.markdown(link4,unsafe_allow_html=True) 

    st.write("6. Sales report") 
    link6 ="https://teams.microsoft.com/l/entity/1c4340de-2a85-40e5-8eb0-4f295368978b/Home?context=%7B%22subEntityId%22%3A%22https%253A%252F%252Fapp.powerbi.com%252Fgroups%252Fme%252Freports%252F5ce5ba78-d626-4843-8326-2ec086f1a0ed%252FReportSection%253Faction%253DOpenReport%2526pbi_source%253DMSTeams%22%7D"
    st.markdown(link6,unsafe_allow_html=True) 
    

if nav == "Visualization":
    st.write("Visualization")  
    df = pd.read_csv("./data_behavior_pattern/train.csv")

    f1 = plt.figure(figsize=(10,6))
    plt.title("Ages Frequency")
    sns.axes_style("dark")
    sns.violinplot(y=df["age"])
    st.pyplot(f1)


    st.markdown("<h5 style='text-align: center;color: Green; text-decoration: underline;'> Gender wise Users</h5>", unsafe_allow_html=True)

    st.write("0 : Male")
    st.write("1 : Female")
    genders = df.gender.value_counts()
    sns.set_style("darkgrid")
    f4 = plt.figure(figsize=(10,4))
    sns.barplot(x=genders.index, y=genders.values)
    st.pyplot(f4)


    st.markdown("<h5 style='text-align: center;color: Green; text-decoration: underline;'> Age wise Users</h5>", unsafe_allow_html=True)

    age18_25 = df.age[(df.age <= 25) & (df.age >= 18)]
    age26_35 = df.age[(df.age <= 35) & (df.age >= 26)]
    age36_45 = df.age[(df.age <= 45) & (df.age >= 36)]
    age46_55 = df.age[(df.age <= 55) & (df.age >= 46)]
    age55above = df.age[df.age >= 56]

    x = ["18-25","26-35","36-45","46-55","55+"]
    y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

    
    f5 = plt.figure(figsize=(15,6))
    sns.barplot(x=x, y=y, palette="rocket")
    plt.title("Number of Customer and Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Customer")
    st.pyplot(f5)


    ss1_20 = df["steps100"][(df["steps100"] >= 1) & (df["steps100"] <= 100)]
    ss21_40 = df["steps100"][(df["steps100"] >= 100) & (df["steps100"] <= 200)]
    ss41_60 = df["steps100"][(df["steps100"] >= 200) & (df["steps100"] <= 300)]
    ss61_80 = df["steps100"][(df["steps100"] >= 300) & (df["steps100"] <= 400)]
    ss81_100 = df["steps100"][(df["steps100"] >= 4000)]

    ssx = ["1-1000", "1000-2000", "2000-3000", "3000-4000", "4000--"]
    ssy = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

    st.markdown("<h5 style='text-align: center;color: Green; text-decoration: underline;'>Customers vs Steps taken and Burnt Calories</h5>", unsafe_allow_html=True)

    f6 = plt.figure(figsize=(15,6))
    sns.barplot(x=ssx, y=ssy, palette="nipy_spectral_r")
    plt.title("Steps taken")
    plt.xlabel("Score")
    plt.ylabel("Number of Customer")
    # plt.show()
    st.pyplot(f6)


    ai0_30 = df["burn_calories100"][(df["burn_calories100"] >= 0) & (df["burn_calories100"] <= 1000)]
    ai31_60 = df["burn_calories100"][(df["burn_calories100"] >= 1000) & (df["burn_calories100"] <= 2000)]
    ai61_90 = df["burn_calories100"][(df["burn_calories100"] >= 2001) & (df["burn_calories100"] <= 3000)]
    ai91_120 = df["burn_calories100"][(df["burn_calories100"] >= 3001) & (df["burn_calories100"] <= 4000)]
    ai121_150 = df["burn_calories100"][(df["burn_calories100"] >= 4001)]

    aix = ["1-1000", "1000-2000", "2000-3000", "3000-4000", "4000--"]
    aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

    f7 =plt.figure(figsize=(15,6))
    sns.barplot(x=aix, y=aiy, palette="Set2")
    plt.xlabel("Burnt calories")
    plt.ylabel("Number of People")
    st.pyplot(f7)


    #BMI Visualization
    label_encoder = preprocessing.LabelEncoder()

    df = pd.read_csv("data_behavior_pattern/train.csv")
    df1 = df[["age","height","weight","activity"]]
    df1['label']= label_encoder.fit_transform(df1['activity'])
    df1['bmi'] = df1['weight']/df1['height']
    df1['bmi'] = df1['bmi']/df1['height']
    df1['bmi'] = df1['bmi']*10000
    # df1
    df2=df1


    st.markdown("<h5 style='text-align: center;color: Green; text-decoration: underline;'> Customer Vs BMI Report</h5>", unsafe_allow_html=True)

    '''

    1=Weak: 16<BMI<18.5

    2=Normal: 18.5<BMI<24.9

    3= Overweight:25<BMI<29.9

    4=Obesity:30<BMI 34.9

    '''
    age16_18 = df2.bmi[(df2.bmi <= 18.5) & (df2.bmi >= 16)]
    age18_24 = df2.bmi[(df2.bmi <= 24.9) & (df2.bmi >= 18.5)]
    age25_29 = df2.bmi[(df2.bmi <= 29.9) & (df2.bmi >= 25)]
    age30_35 = df2.bmi[(df2.bmi <= 34.9) & (df2.bmi >= 30)]

    x = ["16<BMI<18.5","18.5<BMI<24.9","25<BMI<29.9","30<BMI<34.9"]
    y = [len(age16_18.values),len(age18_24.values),len(age25_29.values),len(age30_35.values)]

    f9=plt.figure(figsize=(15,6))
    sns.barplot(x=x, y=y, palette="rocket")
    plt.title("Number of Customer and bmi")
    plt.xlabel("Bmi")
    plt.ylabel("Number of Customer")
    # plt.show()
    st.pyplot(f9)


if nav == "Clustering":

    st.write("Clustering")

    df = pd.read_csv("./data_behavior_pattern/train.csv")
    df1 =df[['age','steps100','burn_calories100']]
    st.dataframe(df1)

    wcss = []
    for k in range(1,11):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(df1)
        wcss.append(kmeans.inertia_)
    f2 = plt.figure(figsize=(12,6))    
    plt.grid()
    plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
    plt.xlabel("K Value")
    plt.xticks(np.arange(1,11,1))
    plt.ylabel("WCSS")
    # plt.show()
    st.pyplot(f2)
    

    title = st.text_input('Number of Cluster')
    # st.write('The current movie title is', int(title))

    if title != '':
        nc = 1
        nc = int(title)

        df1.rename(columns = {'age':'Age','steps100':'Steps Taken','burn_calories100':'Burnt Calories'}, inplace = True)
        km = KMeans(n_clusters= nc)
        clusters = km.fit_predict(df1)
        df1["label"] = clusters

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df1.Age[df1.label == 0], df1["Steps Taken"][df1.label == 0], df1["Burnt Calories"][df1.label == 0], c='blue', s=60)
        ax.scatter(df1.Age[df1.label == 1], df1["Steps Taken"][df1.label == 1], df1["Burnt Calories"][df1.label == 1], c='red', s=60)
        ax.scatter(df1.Age[df1.label == 2], df1["Steps Taken"][df1.label == 2], df1["Burnt Calories"][df1.label == 2], c='green', s=60)
        ax.scatter(df1.Age[df1.label == 3], df1["Steps Taken"][df1.label == 3], df1["Burnt Calories"][df1.label == 3], c='orange', s=60)
        ax.scatter(df1.Age[df1.label == 4], df1["Steps Taken"][df1.label == 4], df1["Burnt Calories"][df1.label == 4], c='purple', s=60)
        ax.view_init(30, 185)
        plt.xlabel("Age")
        plt.ylabel("Steps Taken")
        ax.set_zlabel('Burnt Calories')
        # plt.show()
        st.pyplot(fig)


if nav == "Sales Prediction":
    df = pd.read_csv("./data_sales/sales_part_1.csv")
    st.write("Products")

    t = st.selectbox("What kind of Graph ? ",{"Product-1":6,"Product-2":7,'Product-3':9,
        "Product-4":12,"Product-5":13,"Product-6":14,"Product-7":18})

    if t == "Product-1":
        p_no=6
    if t == "Product-2":
        p_no=7
    if t == 'Product-3':
        p_no=9
    if t == "Product-4":
        p_no=12
    if t == "Product-5":
        p_no=13
    if t == "Product-6":
        p_no=14
    if t == "Product-7":
        p_no=18

    if p_no != '':
        wearables_df = df.loc[df['product_le'] == p_no]
        wearables_df = wearables_df[['day_v','total_sales']]
        wearables_df = wearables_df.groupby('day_v')['total_sales'].sum().reset_index()
        print(wearables_df.head())
        wearables_df = wearables_df.set_index('day_v')
        wearables_df.index = pd.to_datetime(wearables_df.index)
        y = wearables_df['total_sales'].resample('MS').mean()

        yyy = []
        for i in y:
            yyy.append(i)
        aa = plt.figure(figsize = (10,5))
        plt.scatter(list(y.keys()),yyy)
       
        st.pyplot(aa)    

        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit()


                except:
                    continue
                    
        mod = sm.tsa.statespace.SARIMAX(y,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 0, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        results = mod.fit()

        pred = results.get_prediction(start=pd.to_datetime('2019-08-01'), dynamic=False)
        pred_ci = pred.conf_int()

        ax = y['2019':].plot(label='observed')
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=1, figsize=(14, 7))

        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.4)

        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        plt.legend()
        st.pyplot()
        # plt.show()

        y_forecasted = pred.predicted_mean


        y_forecasted = pred.predicted_mean
        y_truth = y['2019-08-01':]

        # Compute the mean square error
        mse = ((y_forecasted - y_truth) ** 2).mean()


        pred_uc = results.get_forecast(steps=6)
        pred_ci = pred_uc.conf_int()

        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Wearables Sales')
        plt.legend()
        st.pyplot()

        st.write(pred_uc.predicted_mean)
    

if nav == "Team":
    
    st.markdown("<h2 style='text-align: center;'>Team 9 AI -Immortals</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Urvi Shah (Team Lead)</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Soumendu Mukherjee (Team Lead)</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Sharique Seraj</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Asadullah Khan</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Omkar Khadilkar</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Ramcharan Naidu Palle</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Velmurugan</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;color: grey;'>Mentor : BIKASH DASH</h5>", unsafe_allow_html=True)

