import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'text'])
    return df

def train_model():
    df = load_data()
    X = df['text']
    y = df['label']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Simpan model dan vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def main():
    st.title("Aplikasi Deteksi Spam SMS Sederhana")
    menu = ["Train Model", "Prediksi Spam"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Train Model":
        if st.button("Train"):
            train_model()
            st.success("Model berhasil dilatih dan disimpan!")
    elif choice == "Prediksi Spam":
        model, vectorizer = load_model()
        input_sms = st.text_area("Masukkan pesan SMS:")
        if st.button("Prediksi"):
            if input_sms:
                input_vec = vectorizer.transform([input_sms])
                prediction = model.predict(input_vec)[0]
                st.write(f"Hasil prediksi: **{prediction.upper()}**")
            else:
                st.warning("Silakan masukkan pesan SMS.")

if __name__ == '__main__':
    main()
