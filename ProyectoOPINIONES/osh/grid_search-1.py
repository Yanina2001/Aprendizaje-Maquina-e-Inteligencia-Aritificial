from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class data_set_polarity:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class data_set_attraction:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

def train(corpus, y_train):
	classifiers_name = ['LR']
	classifiers = [LogisticRegression()]

	best_score = 0
	for clf_name, clf in zip(classifiers_name, classifiers):
		if clf_name == 'LR':
			parameters =  [{'clf__solver': ["sag"], 'clf__C': [1.8,1.81,1.79], 'clf__multi_class': ['ovr'], 'clf__penalty': ['l2'], 'clf__max_iter': [10000]}]

		pipeline = Pipeline([
							('clf', clf)
							])
		#~ print (pipeline)
		#~ cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=2)
		cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=5)
		cv.fit(corpus, y_train)
		
		print ("Classifier: " + clf_name)
		cv_best_score = cv.best_score_
		print("\nBest score: {:0.3f}".format(cv_best_score))
		print("\nBest parameters set:")
		best_parameters = cv.best_estimator_.get_params()
		for dic in parameters:
			for param_name in sorted(dic.keys()):
				print("\t{}: {}".format(param_name, best_parameters[param_name]))
		
		if 	cv_best_score > best_score:
			best_score = cv_best_score
			best_classifier = clf_name
			best_params = best_parameters
			used_params = parameters
			best_model = cv

	print ("*********************************************************")
	print ("                      RESUMEN ENTRENAMIENTO              ")
	print ("*********************************************************")
	print ("Best classifier: " + best_classifier)
	print ("Best parameters:  " )
	for dic in used_params:
		for param_name in sorted(dic.keys()):
			print("\t{}: {}".format(param_name, best_params[param_name]))
	print ("Best accuracy: " + str(best_score))
		
	return (best_model)


def test(model, data_test,y_test):
	print ("*********************************************************")
	print ("         PREDICCIÓN EN EL CONJUNTO DE PRUEBA             ")
	print ("*********************************************************")
	y_predictions = model.predict(data_test)
	print (y_predictions)
	print (y_test)
	report = classification_report(y_test, y_predictions)
	print (report)

	ConfusionMatrixDisplay.from_predictions(y_test, y_predictions)
	plt.show()	



if __name__=='__main__':
	corpus_file = open ('corpus_attraction.pkl','rb')
	corpus_attraction = pickle.load(corpus_file)
	corpus_file.close()

	corpus_file = open ('corpus_polarity.pkl','rb')
	corpus_polarity = pickle.load(corpus_file)
	corpus_file.close()

	vectorizador_binario = CountVectorizer(binary=True)
	vectorizador_binario_fit = vectorizador_binario.fit(corpus_attraction.X_train)
	X_train = vectorizador_binario_fit.transform(corpus_attraction.X_train)
	y_train = corpus_attraction.y_train

	X_test = vectorizador_binario_fit.transform(corpus_attraction.X_test)
	y_test = corpus_attraction.y_test

	# df1 = pd.read_csv("X_train.csv", sep=',', engine='python')
	# df2 = pd.read_csv("y_train.csv", sep=',', engine='python')
	# df3 = pd.read_csv("X_test.csv", sep=',', engine='python')
	# df4 = pd.read_csv("y_test.csv", sep=',', engine='python')

		
	model = train(X_train, y_train)

	test(model, X_test, y_test)