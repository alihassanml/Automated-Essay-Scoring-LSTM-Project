import numpy as np
import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
model = load_model('model.h5')

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

max_len = 1055

def predict(text):
    tokenize = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([tokenize], maxlen=max_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted = np.argmax(np.array(predicted), axis=1)
    return predicted

def main():
    st.title("Automated Essay Scoring System")
    
    text = st.text_area("Enter your essay text here:")
    
    if st.button("Predict"):
        if text:
            prediction = predict(text)
            st.success(f"Predicted Score: {prediction[0]}")
        else:
            st.error("Please enter some text to predict.")

if __name__ == "__main__":
    main()
