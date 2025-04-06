import streamlit as st

st.title("📘 Logistic Regression")
import joblib
import numpy as np

# Load the logistic regression model
model = joblib.load("models/logi.pkl")
option = st.radio("Choose an option", [" Predict Result", " Know the Model Logic"])

# Option 1: Predict Result
if option == " Predict Result":
    st.subheader("🎓 Will the student Pass or Fail?")
    st.write("Enter two grade scores (e.g., internal marks or assessments) to predict if the student will pass (1) or fail (0).")

    grade1 = st.number_input("Enter Grade 1:", min_value=0.0, max_value=100.0, value=60.0)
    grade2 = st.number_input("Enter Grade 2:", min_value=0.0, max_value=100.0, value=60.0)

    if st.button("🔍 Predict"):
        input_data = np.array([[grade1, grade2]])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Predicted: PASS (Confidence: {proba:.2f})")
        else:
            st.error(f"❌ Predicted: FAIL (Confidence: {1 - proba:.2f})")

# Option 2: Know the Model Logic
elif option == " Know the Model Logic":
    st.subheader(" Behind the Logistic Regression Model")

    st.markdown("""
    ### 🔍 Model Info:
    - **Model**: Logistic Regression
    - **Input Features**: `grade1`, `grade2`
    - **Target**: `0` = Fail, `1` = Pass
    - **Accuracy**: ~80% on test data

    ### 🛠 Training Steps:
    """)
    
    st.code("""
# 1. Clean dataset
df['label'] = df['label'].str.replace(';', '', regex=False).astype(int)

# 2. Feature selection
X = df[['grade1', 'grade2']]
y = df['label']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Save model
joblib.dump(model, 'models/prepare_model3.pkl')
    """, language="python")

    st.markdown("""
    ### ✅ Why Logistic Regression?
    - Interpretable and fast for binary classification
    - Gives class probabilities
    - Works well with small and clean datasets

    """)
    st.markdown("## Summary Output Screen")
    st.image("images/logistic.png", caption="Predicted Price Based on Input", use_column_width=True)

