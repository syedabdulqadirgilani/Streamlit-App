import streamlit as st # pip install streamlit
import numpy as np # pip install numpy
import pandas as pd # pip install pandas
import matplotlib.pyplot as plt # pip install matplotlib
import seaborn as sns # pip install seaborn
from sklearn.linear_model import LogisticRegression # pip install sklearn
from sklearn.model_selection import train_test_split # pip install sklearn
from sklearn.metrics import accuracy_score # pip install sklearn
df=sns.load_dataset('iris') # loading the dataset
df=df.dropna() # dropping the null values
st.title('Iris Flower Prediction App') # title of the app
st.write('This app predicts the **Iris flower** type!') # description of the app
st.sidebar.header('User Input Parameters') # sidebar header
def user_input_features(): # function to take user input
    sepal_length=st.sidebar.slider('Sepal length',4.3,7.9,5.4) # slider for sepal length
    sepal_width=st.sidebar.slider('Sepal width',2.0,4.4,3.4) # slider for sepal width
    petal_length=st.sidebar.slider('Petal length',1.0,6.9,1.3) # slider for petal length
    petal_width=st.sidebar.slider('Petal width',0.1,2.5,0.2) # slider for petal width
    data={'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width} # storing the user input in a dictionary
    features=pd.DataFrame(data,index=[0]) # converting the dictionary into a dataframe
    return features # returning the dataframe
df_new=user_input_features() # calling the function
st.subheader('User Input Parameters') # subheader
st.write(df_new) # displaying the dataframe
X=df[['sepal_length','sepal_width','petal_length','petal_width']] # features
y=df['species'] # target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42) # splitting the dataset
model=LogisticRegression() # model
model.fit(X_train,y_train) # fitting the model
y_pred=model.predict(X_test) # predicting the target
st.subheader('Class labels and their corresponding index number') # subheader
st.write(df['species'].unique()) # displaying the unique values of the target
st.subheader('Model Test Accuracy Score:') # subheader
st.write(accuracy_score(y_test,y_pred)) # displaying the accuracy score
prediction=model.predict(df_new) # predicting the target for the user input
st.subheader('Predicted Target') # subheader
st.write(prediction) # displaying the predicted target
st.subheader('Prediction Probability') # subheader
st.write(model.predict_proba(df_new)) # displaying the prediction probability
sns.heatmap(df.corr(),annot=True,cmap='viridis') # heatmap
st.set_option('deprecation.showPyplotGlobalUse', False) 
st.pyplot() # Streamlit function to display the plot
sns.boxplot(x='species',y='petal_length',data=df) # boxplot
st.set_option('deprecation.showPyplotGlobalUse', False) 
st.pyplot() # Streamlit function to display the plot