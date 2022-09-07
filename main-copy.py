from tkinter import E
from flask import Flask, render_template, request, url_for, jsonify, Response
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from  Video.camera import VideoCamera

app = Flask(__name__)

cNEU = pickle.load( open("Text\pickles\cNEU_project.p", "rb"))
cEXT = pickle.load( open("Text\pickles\cEXT_project (1).p", "rb"))
cAGR = pickle.load( open("Text\pickles\cAGR_project (1).p", "rb"))
cCON = pickle.load( open("Text\pickles\cCON_project (1).p", "rb"))
cOPN = pickle.load( open("Text\pickles\cOPN_project (1).p", "rb"))

with open("Text/pickles/tfidf_vectorizer_project (1).p", 'rb') as f:
    tfidf_transformer = pickle.load(f)

with open('Text/pickles/bow_vectorizer_project.p', 'rb') as f1:
    bow_transformer = pickle.load(f1)

def predict_personality(text):
    sentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = tfidf_transformer.transform(sentences)
    text_vector_32 = bow_transformer.transform(sentences)
    EXT = cEXT.predict(text_vector_31)
    NEU = cNEU.predict(text_vector_32)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/text.html')
def hello_world2():
    return render_template("text.html")

@app.route('/text.html', methods=['GET','POST'])
def pred():
    if request.method == 'POST':
        message = request.form['message']
        sentences = re.split("(?<=[.!?]) +", message)
        text_vector_31 = tfidf_transformer.transform(sentences)
        text_vector_32 = bow_transformer.transform(sentences)
        EXT = cEXT.predict(text_vector_31)
        NEU = cNEU.predict(text_vector_32)
        AGR = cAGR.predict(text_vector_31)
        CON = cCON.predict(text_vector_31)
        OPN = cOPN.predict(text_vector_31)
        pred = [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]

        return render_template("text.html", predictions=pred, mes=message)

@app.route('/text_predict.html', methods=['GET','POST'])
def hello_world3():
    if request.method == 'POST':
        message = request.form['message']
        sentences = re.split("(?<=[.!?]) +", message)
        text_vector_31 = tfidf_transformer.transform(sentences)
        text_vector_32 = bow_transformer.transform(sentences)
        EXT = cEXT.predict(text_vector_31)
        NEU = cNEU.predict(text_vector_32)
        AGR = cAGR.predict(text_vector_31)
        CON = cCON.predict(text_vector_31)
        OPN = cOPN.predict(text_vector_31)
        pred = [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]

        return render_template("text_predict.html", predictions=pred, mes=message)


@app.route('/video_start.html')
def hello_video2():
    return render_template("video_start.html")


@app.route('/vid_index.html', methods=['GET','POST'])
def hello_video():
    return render_template("vid_index.html")

def gen(camera):
    
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)