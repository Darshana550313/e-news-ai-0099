# Required packages: pip install pytesseract streamlit google-api-python-client transformers torch requests pyttsx3
#!pip install pipeline
import pytesseract
from PIL import Image
import os
from transformers import pipeline
from googleapiclient.discovery import build
import streamlit as st
from io import BytesIO
import pyttsx3
import platform

#  Dynamic Tesseract path for Windows/Linux
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Google API setup
api_key = "AIzaSyAKka_SPj4x6gbl-IM2NdxSd6GZKmT2Hlg"  
youtube = build('youtube', 'v3', developerKey=api_key)

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#  Zero-shot Fake News Classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#  TTS function
def speak_text(text):
    if platform.system() == "Windows":
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.error(f"TTS failed: {e}")
    else:
        st.info("TTS is not supported on this platform.")

# ? YouTube search
def search_youtube(query):
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=3
    )
    response = request.execute()
    video_data = []
    for item in response['items']:
        video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        thumbnail = item['snippet']['thumbnails']['default']['url']
        title = item['snippet']['title']
        video_data.append((title, video_url, thumbnail))
    return video_data

# ? Streamlit GUI
st.title("E-NEWS AI - Intelligent News from Image")

st.markdown("###  Upload a news image to extract, summarize and verify the news.")

uploaded_file = st.file_uploader("Upload a News Image", type=["jpg", "png", "jpeg"])
language = st.text_input("Enter OCR Language Code (e.g., eng, hin)", value="eng")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ? OCR
    headline = pytesseract.image_to_string(image, lang=language)
    st.subheader("Extracted Text")
    st.write(headline)

    # ? Summarization
    try:
        summary = summarizer(headline, max_length=50, min_length=25, do_sample=False)
        summary_text = summary[0]['summary_text']
    except Exception as e:
        summary_text = "Could not summarize the text."
        st.warning(f"Summarization failed: {e}")

    st.subheader("Summary")
    st.write(summary_text)

    # ? Fake News Detection using Zero-shot Classification
    st.subheader("Fake News Detection")
    try:
        labels = ["real news", "fake news"]
        prediction = classifier(summary_text, candidate_labels=labels)
        scores = dict(zip(prediction['labels'], prediction['scores']))

        if scores["real news"] > scores["fake news"]:
            st.success(f"This news appears to be **REAL** with {scores['real news']:.2%} confidence.")
        else:
            st.error(f"This news appears to be **FAKE** with {scores['fake news']:.2%} confidence.")
    except Exception as e:
        st.warning(f"Fake news detection failed: {e}")

    # ? TTS Audio
    st.subheader("Listen to Text and Summary")
    if st.button("Play Extracted Text"):
        speak_text(headline)
    if st.button("Play Summary"):
        speak_text(summary_text)

    # ? YouTube Section
    st.subheader("Related YouTube Videos")
    try:
        for title, url, thumb in search_youtube(summary_text):
            st.image(thumb, width=120)
            st.markdown(f"[{title}]({url})")
    except Exception as e:
        st.warning(f"Could not fetch YouTube videos: {e}")

    # ? Download Summary
    st.subheader("Download summary ")
    buffer = BytesIO()
    buffer.write(headline.encode())
    buffer.write(b"\n\nSummary:\n")
    buffer.write(summary_text.encode())
    st.download_button(label="Download Summary", data=buffer.getvalue(), file_name="summary.txt", mime="text/plain")

else:
    st.info("Please upload an image above to get started.")
