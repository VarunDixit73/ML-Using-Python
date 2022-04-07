# fapp -> shortcut from happy flasker extension
from statistics import mode
from telnetlib import BM
from flask import Flask, render_template,request
import numpy as np
from joblib import load
import os

app = Flask(__name__)

def load_clf_model():
    print(os.listdir())
    filepath = 'app_2/clf_ap.pkl'
    return load(filepath)

def predict(Pregnancies,Glucose,BMI,pedigree):
    userinp = np.array([[Pregnancies,Glucose,BMI,pedigree]]) 
    model_dict = load_clf_model()
    x = model_dict.get('scaler').transform(userinp)
    p = model_dict.get('classifier').predict(x)
    if p[0] == 0:
        return "Not Dibatic"
    else:
        return "Dibatic"

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=="POST":
        form = request.form
        Pregnancies = int(form.get('Pregnancies'))
        Glucose = int(form.get('Glucose'))
        BMI = float(form.get('BMI'))
        pedigree = float(form.get('pedigree'))
        result = predict(Pregnancies,Glucose,BMI,pedigree)
        return render_template('index.html',Pregnancies=Pregnancies, Glucose=Glucose, BMI=BMI, pedigree=pedigree,result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
 