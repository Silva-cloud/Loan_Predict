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


    if(LR_prediction[0]== 1):
        Labeled_LR_prediction= 'Yes'
    elif(LR_prediction[0]== 0):
        Labeled_LR_prediction= 'No'

    if(DT_prediction[0]== 1):
        Labeled_DT_prediction= 'Yes'
    elif(DT_prediction[0]== 0):
        Labeled_DT_prediction= 'No'

    if(NB_prediction[0]== 1):
        Labeled_NB_prediction= 'Yes'
    elif(NB_prediction[0]== 0):
        Labeled_NB_prediction= 'No'

    return render_template(
        'index.html', 
        Logistic_Regression_result= 'نتيجة توقع نموذج ال Logisitic Regression هي {}'.format(Labeled_LR_prediction),
        Decision_Tree_result= 'نتيجة توقع نموذج ال Decision Tree هي {}'.format(Labeled_DT_prediction),
        Naive_Bias_result= 'نتيجة توقع نموذج ال Naive Bias هي {}'.format(Labeled_NB_prediction))

if __name__ == '__main__':
    app.run(debug=True)
