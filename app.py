from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

app=Flask(__name__)
Swagger(app)

mnb = pickle.load(open('Naive_Bayes_model.pkl','rb'))
countVect = pickle.load(open('tfidfvectcount.pkl','rb'))

lr=pickle.load(open('LogisticModel.pkl','rb'))

tokenizer = pickle.load(open('tokenizer.pkl','rb'))


yaml_file = open('lstm2.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = tensorflow.keras.models.model_from_yaml(loaded_model_yaml)
loaded_model.load_weights("lstm2.h5")





@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        vect = countVect.transform(data).toarray()
        my_prediction = mnb.predict(vect)
    return render_template('result.html',prediction = my_prediction)

@app.route('/logisticpredict',methods=['POST'])
def logisticpredict():

    if request.method == 'POST':
        Reviews1 = request.form['Reviews1']
        data1 = [Reviews1]
        my_prediction1 = lr.predict(data1)
    return render_template('result.html',prediction = my_prediction1)

@app.route('/LSTMpredict',methods=['POST'])
def LSTMpredict():

    if request.method == 'POST':
        Reviews2 = request.form['Reviews2']
        data2 = [Reviews2]
        sequences_t = tokenizer.texts_to_sequences([data2])
        X_train_s = sequence.pad_sequences(sequences_t, maxlen=100)
        my_prediction2 = np.argmax(loaded_model.predict(X_train_s))

    return render_template('result.html',prediction = my_prediction2)

if __name__ == '__main__':
    app.run(debug=True)
    