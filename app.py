from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def main():
    """
    function to render index.html and its attributes
    """
    if request.method == 'POST':
        pred = predict_iris()
        prediction = pred.replace("-", " ")  # replace '-' with ' '
    else:
        prediction = ""
    return render_template('index.html', prediction=prediction)


def request_values():
    """
    function to get values from the html form
    """
    values = [float(request.form.get('petal_length')), float(request.form.get('petal_width')),
              float(request.form.get('sepal_length')), float(request.form.get('sepal_width'))]
    return values


def predict_iris():
    """
    function to get the values and to predict the iris species
    """
    model = pickle.load(open("models/iris_classification.model", 'rb'))  # load saved model
    if request.method == 'POST':
        prediction = model.predict([request_values()])
        pred = prediction.tolist()  # convert numpy array to list
        for i in pred:
            return i


if __name__ == '__main__':
    app.run()
