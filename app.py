import numpy as np
from flask import Flask, request, render_template
import pickle
app = Flask(__name__)

LRmodel= pickle.load(open('models/LRmodel','rb'))
DTmodel= pickle.load(open('models/DTmodel','rb'))
NBmodel= pickle.load(open('models/NBmodel','rb'))


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    int_data=[int(x) for x in request.form.values()]
    data= [np.array(int_data)]
    LR_prediction = LRmodel.predict(data)
    DT_prediction = DTmodel.predict(data)
    NB_prediction = NBmodel.predict(data)
    return render_template(
        'index.html', 
        prediction_text= 
        'Logistic Regression model says {}\n'.format(LR_prediction)+
        'Decision Tree model says {}\n'.format(DT_prediction)+
        'Naive Bayes model says {}'.format(NB_prediction))

if __name__ == '__main__':
    app.run(debug=True)
