from flask import Flask, escape, request, render_template
import pickle

vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/index', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['newsbox'])
        predict = model.predict(vector.transform([news]))[0]
        return render_template("index.html", prediction_text="News headline is {}".format(predict))
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run()