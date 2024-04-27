import streamlit as st
import numpy as np
import whisper
import os
from audiorecorder import audiorecorder
from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

whisperModel = whisper.load_model("base")

def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

def inference(audio):
    with open ('tempSoundFile.mp3', 'wb') as myFile:
        myFile.write(audio)

    audio = whisper.load_audio('tempSoundFile.mp3')
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(whisperModel.device)

    _, probs = whisperModel.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisperModel, mel, options)

    os.remove("tempSoundFile.mp3")

    return result.text, lang

def extract_emotion(wav_audio_data):
    raw_text2, lang = inference(wav_audio_data)
    sentimentData = analyze_sentiment(raw_text2)

    sentimentname = ''
    sentimentval = ''

    for key, val in sentimentData.items():
        sentimentname = key
        sentimentval = val

    return raw_text2, sentimentname, sentimentval, lang

def main():
    st.title("Voice Emotion Detection")
    st.subheader("Detect Emotions In Voice")

    audio = audiorecorder("Click to record", "Click to stop recording")
    wav_audio_data = audio.export().read()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')

    submit_text = st.button(label='Submit')

    if submit_text:
        raw_text2, prediction, probability, lang = extract_emotion(wav_audio_data)

        st.success("Original Text")
        st.write(raw_text2)

        st.success("Prediction")
        st.write("{}".format(prediction))
        st.write("Confidence: {}".format(np.max(probability)))



if __name__ == '__main__':
    main()
