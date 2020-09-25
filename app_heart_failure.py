from flask import Flask,render_template,request
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pd.read_pickle('rfc_model_heart.pickle')

@app.route('/')	
def home():
	
	df=pd.read_csv('heartfailure.csv')

	lis=[]
	X=df.drop('DEATH_EVENT',axis=1)
	x=X.columns
	for i in x:
		lis.append(i) 
	

	return render_template('home.html', len=len(lis), lis=lis)



@app.route('/predict', methods=['POST','GET'])
def predict():
	df=pd.read_csv('heartfailure.csv')

	lis=[]
	X=df.drop('DEATH_EVENT',axis=1)
	x=X.columns
	

	if request.method=='POST':
		for i in x:
			lis.append(request.form[f'{i}'])

	
	chances=model.predict([lis])


	return render_template('result.html',prediction_text=f'and the prediction is : {chances[0]}' )

if __name__=='__main__':
	app.run(debug=True)