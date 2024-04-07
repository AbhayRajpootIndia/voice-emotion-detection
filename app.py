import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import whisper
import os
from audiorecorder import audiorecorder

import joblib

whisperModel = whisper.load_model("base")

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

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

    return result.text

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    audio = audiorecorder("Click to record", "Click to stop recording")
    print(audio)
    wav_audio_data = audio.export().read()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')

    submit_text = st.button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        raw_text2 = inference(wav_audio_data)

        prediction = predict_emotions(raw_text2)
        probability = get_prediction_proba(raw_text2)

        with col1:
            st.success("Original Text")
            st.write(raw_text2)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)






if __name__ == '__main__':
    main()
