import streamlit as st

st.set_page_config(page_title="Play with Predictions", layout="centered")

st.title("Play with Predictions")
st.subheader("Choose what you'd like to explore:")

st.page_link("pages/dashboard.py", label=" Dashboard & Visualization")
st.page_link("pages/linear.py", label="Learn about Linear Regression")
st.page_link("pages/naives.py", label=" Learn about Multinomial Naive Bayes ")
st.page_link("pages/multilinear.py", label=" Learn about Multiple Linear Regression")
