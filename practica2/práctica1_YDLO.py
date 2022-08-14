"""
Alumno: De Luna Ocampo Yanina
Fecha: 01-03-2022
"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import sys
import pickle 

#Modificamos las impresiones de dataframes
np.set_printoptions(threshold=sys.maxsize, suppress=True)

class validation_set:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test

class data_set:
	def __init__(self, validation_set, test_set):
		self.validation_set = validation_set
		self.test_set = test_set

class models:
	def __init__(self, folds_3, folds_5, folds_10):
		self.folds_3 = folds_3
		self.folds_5 = folds_5
		self.folds_10 = folds_10
        
def generate_train_test(file_name,folds):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['RainTomorrow'],axis=1).values   
	y = df['RainTomorrow'].values
	
	#80% para entrenamiento y el 20% para pruebas
	X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.2, shuffle = True)
	
	#Crea pliegues para la validación cruzada
	validation_sets = []
	kf = KFold(n_splits=folds)
	for train_index, test_index in kf.split(X_train):
		#~ print("TRAIN:", train_index, "\n",  "TEST:", test_index)
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		#~ #Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
	
	#Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)	
	
	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set) 
	
	return (my_data_set)
	
if __name__=='__main__':
    
    #Empezamos con k = 3
	my_data_set_3_folds = generate_train_test('weatherAUS.csv',3)
	my_data_set_5_folds = generate_train_test('weatherAUS.csv',5)
	my_data_set_10_folds = generate_train_test('weatherAUS.csv',10)
	
	print (my_data_set_3_folds.test_set.X_test)
	#~ print(type(my_data_set.test_set.X_test))
	#~ print ('\n----------------------------------------------------------------------------------\n')
	
	#Guarda el dataset en formato csv
	np.savetxt("data_test_3folds.csv", my_data_set_3_folds.test_set.X_test, delimiter=",", fmt="%s",
           header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday")
	np.savetxt("target_test_3folds.csv", my_data_set_3_folds.test_set.y_test, delimiter="," , fmt="%s",
           header="RainTomorrow", comments="")


	i = 1
	for val_set in my_data_set_3_folds.validation_set:
		np.savetxt("data_validation_train_"+ str(3) + "_" + str(i) + "_" + ".csv", val_set.X_train, delimiter=",", fmt="%s",
           header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", comments="")
		np.savetxt("data_validation_test_" + str(3) + "_" + str(i) + "_" + ".csv", val_set.X_test, delimiter=",", fmt="%s",
           header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday", comments="")
		np.savetxt("target_validation_train_"+ str(3) + "_" + str(i) + "_" + ".csv", val_set.y_train, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		np.savetxt("target_validation_test_" + str(3) + "_" + str(i) + "_" + ".csv", val_set.y_test, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		i += 1
	




    #Seguimos con k=5
	
	print(my_data_set_5_folds.test_set.X_test)
	#Guarda el dataset en formato csv
	np.savetxt("data_test_5folds.csv", my_data_set_5_folds.test_set.X_test, delimiter=",", fmt="%s",
           header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday")
	np.savetxt("target_test_5folds.csv", my_data_set_5_folds.test_set.y_test, delimiter="," , fmt="%s",
           header="RainTomorrow", comments="")


	i = 1
	for val_set in my_data_set_5_folds.validation_set:
		np.savetxt("data_validation_train_"+ str(5) + "_" + str(i) + "_" + ".csv", val_set.X_train, delimiter=",", fmt="%s",
           header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", comments="")
		np.savetxt("data_validation_test_" + str(5) + "_" + str(i) + "_" + ".csv", val_set.X_test, delimiter=",", fmt="%s",
           header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday", comments="")
		np.savetxt("target_validation_train_"+ str(5) + "_" + str(i) + "_" + ".csv", val_set.y_train, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		np.savetxt("target_validation_test_" + str(5) + "_" + str(i) + "_" + ".csv", val_set.y_test, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		i += 1
    





	#Y por último terminamos con el de k=10
    
	print (my_data_set_10_folds.test_set.X_test)
	#~ print(type(my_data_set.test_set.X_test))
	#~ print ('\n----------------------------------------------------------------------------------\n')
	
	#Guarda el dataset en formato csv
	np.savetxt("data_test_10folds.csv", my_data_set_10_folds.test_set.X_test, delimiter=",", fmt="%s",
           header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday")
	np.savetxt("target_test_10folds.csv", my_data_set_10_folds.test_set.y_test, delimiter="," , fmt="%s",
           header="RainTomorrow", comments="")


	i = 1
	for val_set in my_data_set_10_folds.validation_set:
		np.savetxt("data_validation_train_"+ str(10) + "_" + str(i) + "_" + ".csv", val_set.X_train, delimiter=",", fmt="%s",
           header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", comments="")
		np.savetxt("data_validation_test_" + str(10) + "_" + str(i) + "_" + ".csv", val_set.X_test, delimiter=",", fmt="%s",
           header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday", comments="")
		np.savetxt("target_validation_train_"+ str(10) + "_" + str(i) + "_" + ".csv", val_set.y_train, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		np.savetxt("target_validation_test_" + str(10) + "_" + str(i) + "_" + ".csv", val_set.y_test, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		i += 1
		
	#creamos una instancia del objeto models
	model = models(my_data_set_3_folds, my_data_set_5_folds, my_data_set_10_folds)
	#guardamos el objeto en un fichero pickle
	with open('models.pkl', 'wb') as f:
		pickle.dump(model, f)
		f.close()
	#abrimos el fichero pickle
	with open('models.pkl', 'rb') as f:
		pickle_models = pickle.load(f)
		f.close()
	#imprimimos los resultados
	print ("-----------------------------------------------")
	print(pickle_models.folds_3.test_set.X_test)
	
 	#dataset_files = open ('dataset.pkl','wb')
 	#pickle.dump(my_data_set, dataset_file)
 	#dataset_file.close()
 	
 	#dataset_file = open ('dataset.pkl','rb')
 	#my_data_set_pickle = pickle.load(dataset_file)
 	#print ("-----------------------------------------------")
 	#print (my_data_set_pickle.test_set.X_test)