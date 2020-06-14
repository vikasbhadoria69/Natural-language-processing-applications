from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename_rest = 'restaurant-sentiment-mnb-model.pkl'
filename_spam = 'spam-sms-mnb-model.pkl'
classifier_rest = pickle.load(open(filename_rest, 'rb'))
classifier_spam = pickle.load(open(filename_spam, 'rb'))
cv_rest = pickle.load(open('cv-transform-rest.pkl','rb'))
cv_spam = pickle.load(open('cv-transform-spam.pkl','rb'))

app = Flask(__name__,template_folder='template')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/spam')
def spam():
	return render_template('index_spam.html')

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv_spam.transform(data).toarray()
    	my_prediction = classifier_spam.predict(vect)
    	return render_template('result_spam.html', prediction=my_prediction)

@app.route('/resturant')
def resturant():
	return render_template('index_rest.html')

@app.route('/predict_resturant', methods=['POST'])
def predict_resturant():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv_rest.transform(data).toarray()
    	my_prediction = classifier_rest.predict(vect)
    	return render_template('result_rest.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)