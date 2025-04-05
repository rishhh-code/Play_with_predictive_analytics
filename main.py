import streamlit as st

st.set_page_config(page_title="Play with Predictions", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        .title-text {
            text-align: center;
            font-size: 3em;
            color: #3ECF8E;
            font-weight: 700;
        }
        .subtitle-text {
            text-align: center;
            font-size: 1.3em;
            color: #CCCCCC;
            margin-bottom: 60px;
        }
        .link-card {
            background-color: #1f1f1f;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s;
            color: white;
            text-decoration: none;
            font-size: 1.1em;
            display: block;
            margin: 10px;
        }
        .link-card:hover {
            transform: scale(1.03);
            background-color: #292929;
        }
        .footer {
            text-align: center;
            margin-top: 80px;
            color: gray;
        }
        a {
            text-decoration: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown('<div class="title-text">Play with Predictions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Explore machine learning models with real-world use cases</div>', unsafe_allow_html=True)

# First row
with st.container():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            """
            <a href="/dashboard">
                <div class="link-card">📊<br><br><strong>Dashboard & Visualization</strong></div>
            </a>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <a href="/logistic">
                <div class="link-card">📘<br><br><strong>Logistic Regression: Pass/Fail Predictor</strong></div>
            </a>
            """,
            unsafe_allow_html=True
        )

# Second row
with st.container():
    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown(
            """
            <a href="/naives">
                <div class="link-card">✉️<br><br><strong>Naive Bayes: Spam Classifier</strong></div>
            </a>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <a href="/multilinear">
                <div class="link-card">🏠<br><br><strong>Multiple Linear Regression: House Price</strong></div>
            </a>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    """
    <div class="footer">
        “The goal is to turn data into information, and information into insight.” – Carly Fiorina
        <br><br>
        Made with ❤️ by Pookies | 2025
    </div>
    """,
    unsafe_allow_html=True
)
