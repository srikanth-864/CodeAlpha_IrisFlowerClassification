import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)

st.title("Iris Flower Classifier")
sepal_length = st.slider('Sepal length', 4.0, 8.0)
sepal_width = st.slider('Sepal width', 2.0, 5.0)
petal_length = st.slider('Petal length', 1.0, 7.0)
petal_width = st.slider('Petal width', 0.1, 2.5)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = iris.target_names[model.predict(input_data)][0]
st.success(f"The predicted species is: **{prediction}**")