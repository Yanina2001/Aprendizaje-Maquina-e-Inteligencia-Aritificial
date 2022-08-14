from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay

def train(corpus, y_train):
	classifiers_name = ['MLP']
	classifiers = [MLPClassifier()]

	best_score = 0
	for clf_name, clf in zip(classifiers_name, classifiers):
		if clf_name == 'MLP':
			parameters =  [{'hidden_layer_sizes': [(500,), (200,), (100,)],
                            'activation':['relu', 'logistic', 'identity'], 
                            'solver': ['adam', 'sgd'], 
                            'alpha': [0.0001, 0.00001, 0.001],
                            'learning_rate': ['invscaling', 'adaptative'], 
                            'max_iter': [200]}]

		# pipeline = Pipeline([
		# 					('clf', clf)
		# 					])
		#~ print (pipeline)
		#~ cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=2)
		cv = GridSearchCV(MLPClassifier(), param_grid=parameters, n_jobs=-1, cv=5)
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
	print ("                      RESUMEN ENTRENAMIENTO                           ")
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

#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
print ('Cargando datos...')
df = pd.read_csv('mnist_train.csv', sep=',', engine='python')
X= df.drop(['label'],axis=1).values   
y = df['label'].values
# X_test = pd.read_csv('test.csv', sep=',', engine='python')
# X_test = X_test.values
# X_test = df2.drop(['label'],axis=1).values   
# y_test = df2['label'].values
# Separa el corpus cargado en el DataFrame en el 70% para entrenamiento y el 30% para pruebas
print ('Separando los conjuntos de datos...')
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, shuffle = True, random_state=0)


# model = train(X_train, y_train)

# test(model, X_test, y_test)
# for i in range(10):
#     img = X_train[i].reshape((28,28))
#     plt.imshow(img, cmap="Greys")
#     plt.show()
	
clf = MLPClassifier(hidden_layer_sizes=(500,), max_iter=500, random_state = 0, solver = 'lbfgs', learning_rate='invscaling', alpha=0.001, activation='logistic')

print ('Entrenando red neuronal ...')
clf.fit(X_train, y_train)

print ('Predicción de la red neuronal')
y_pred = clf.predict(X_test)

report = classification_report(y_test, y_pred)
print (report)

# print (classification_report(y_test, y_pred))
print("Training set score: %f" % clf.score(X_train, y_train))

#Entrenando con el conjunto de entrenamiento completo
print ('Entrenando red neuronal ...')
clf.fit(X, y)

#Probando con el conjunto de prueba completo
print ('Predicción de la red neuronal')
df = pd.read_csv('mnist_test.csv', sep=',', engine='python')
X_test = df.drop(['label'],axis=1).values
y_test = df['label'].values

print ('Predicción de la red neuronal')
y_pred = clf.predict(X_test)

report = classification_report(y_test, y_pred)
print (report)

for i in range(len(y_test)):
	if(y_test[i]!=y_pred[i]):
		img = X_test[i].reshape((28,28))
		plt.imshow(img, cmap="Greys")
		plt.title('Real: ' + str(y_test[i]) + ' Predicted: ' + str(y_pred[i]), fontsize = 20);
		plt.show()
		
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()