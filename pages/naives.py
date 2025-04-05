import streamlit as st
import joblib
import re
import string

# Load trained model
model = joblib.load("models/spam_detector.pkl")

st.title(" 📘 Multinomial Naive Bayes ")

option = st.radio("Choose an option", [" Play with Spam Mail Detector", " Know Its Logic"])

# Option 1: Play with Spam Detector
if option == " Play with Spam Mail Detector":
    st.subheader("Spam Detector (Multinomial Naive Bayes)")
    st.write("Enter a message to check if it's **Spam** or **Ham**.")

    user_input = st.text_area(" Enter your message:", height=100)

    if st.button(" Predict "):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            proba = model.predict_proba([user_input])[0][1]

            if prediction == 1:
                st.error(f"🚫 It's likely SPAM! (Confidence: {proba:.2f})")
            else:
                st.success(f"✅ It's HAM / Not Spam! (Confidence: {1 - proba:.2f})")
        else:
            st.warning("⚠️ Please enter a message.")

# Option 2: Know the Logic Behind It
elif option == " Know Its Logic":
    st.subheader(" Behind the Spam Detector (Naive Bayes Model)")

    st.markdown("""
    ##  Model Overview:
    - **Model Used**: Multinomial Naive Bayes
    - **Vectorizer**: TF-IDF with top 5000 words
    - **Dataset**: SMS Spam Collection Dataset
    - **Accuracy**: ~94.8% on test set
    - **Cross-Validation Score**: ~95.8%

    ##  Preprocessing & Training Steps
    """)

    st.code("""
# Step 1: Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Step 2: Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(...)

# Step 4: Define the model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(...)),
    ('clf', MultinomialNB())
])

# Step 5: Train the model
pipeline.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 7: Save the model
joblib.dump(pipeline, "models/spam_detector.pkl")
    """, language='python')

    st.markdown("## Training Output:")

    st.image("images/nbmodel.png", caption="Model Accuracy & Report", use_column_width=True)

    st.markdown("""
    ##  Line-by-Line Code Explanation:

    - **`preprocess_text`**: Cleans the text by converting to lowercase, removing digits and punctuation.
    - **`df['label'] = df['label'].map(...)`**: Converts 'ham' to 0 and 'spam' to 1 for binary classification.
    - **`train_test_split(...)`**: Splits data into 80% training and 20% testing.
    - **`TfidfVectorizer(...)`**: Converts words into numerical importance values (Term Frequency-Inverse Document Frequency).
    - **`MultinomialNB()`**: A Naive Bayes model suited for text classification.
    - **`pipeline.fit(...)`**: Trains the model on training data.
    - **`classification_report(...)`**: Evaluates precision, recall, F1 score for both spam and ham.
    - **`joblib.dump(...)`**: Saves the trained model for use in the Streamlit app.

    ##  Why Naive Bayes Works Well:
    - Assumes word independence (bag of words) — surprisingly effective in spam classification!
    - Fast, scalable, and works great with TF-IDF for feature extraction.
    """)

