import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

DATASET_PATH = "SMSSpamCollection"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

@st.cache_data
def load_data():
    # Dataset harus ada di folder yang sama, format tab separated
    df = pd.read_csv(DATASET_PATH, sep='\t', header=None, names=['label', 'text'])
    return df

def train_and_save_model():
    df = load_data()
    X = df['text']
    y = df['label']

    # Vectorizer
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save model and vectorizer
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    # Evaluate
    acc = model.score(X_test, y_test)
    return acc

def load_model_and_vectorizer():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def main():
    st.title("Aplikasi Deteksi Spam SMS Sederhana")

    menu = ["Train Model", "Prediksi Spam"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Train Model":
        st.subheader("Train Model Deteksi Spam")
        if st.button("Mulai Training"):
            acc = train_and_save_model()
            st.success(f"Model berhasil dilatih! Akurasi pada data test: {acc:.2f}")
            st.info("Model dan vectorizer telah disimpan (model.pkl & vectorizer.pkl).")
        st.write("**Pastikan file dataset SMSSpamCollection sudah ada di folder ini.**")

    elif choice == "Prediksi Spam":
        st.subheader("Prediksi Pesan SMS")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            st.warning("Model belum dilatih. Silakan train model terlebih dahulu.")
        else:
            model, vectorizer = load_model_and_vectorizer()
            input_sms = st.text_area("Masukkan pesan SMS yang ingin dicek:", "")
            if st.button("Prediksi"):
                if input_sms.strip() == "":
                    st.warning("Masukkan pesan SMS terlebih dahulu.")
                else:
                    input_vec = vectorizer.transform([input_sms])
                    prediction = model.predict(input_vec)[0]
                    if prediction.lower() == "spam":
                        st.error("Pesan ini terdeteksi sebagai: SPAM")
                    else:
                        st.success("Pesan ini terdeteksi sebagai: HAM (bukan spam)")

if __name__ == "__main__":
    main()
