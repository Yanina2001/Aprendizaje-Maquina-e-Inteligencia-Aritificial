import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import sys
import pickle 

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

def generate_train_test(file_name,folds):
	pd.options.display.max_colwidth = 200			

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['diagnosis',"id"],axis=1).values; y = df['diagnosis'].values
	
	#Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
	X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.2, shuffle = True)
	
	#~ #Crea pliegues para la validacion cruzada
	validation_sets = []
	kf = KFold(n_splits=folds)
	for train_index, test_index in kf.split(X_train):
		#~ print("TRAIN:", train_index, "\n",  "TEST:", test_index)
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		#~ #Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
	
	#~ #Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)	
	
	#~ #Guarda el dataset con los pliegues del conjunto de validacion y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set) 
	
	return (my_data_set)

if __name__=='__main__':
    
    #Para k = 3
	my_data_set_3_folds = generate_train_test('breast-cancer.csv',3)
	
	#Guarda el dataset en formato csv
	np.savetxt("data_test_3folds.csv", my_data_set_3_folds.test_set.X_test, delimiter=",", fmt="%s",
           header="radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst")
	
	np.savetxt("target_test_3folds.csv", my_data_set_3_folds.test_set.y_test, delimiter="," , fmt="%s",
           header="diagnosis", comments="")


	i = 1
	for val_set in my_data_set_3_folds.validation_set:
		np.savetxt("data_validation_train_"+ str(3) + "_" + str(i) + "_" + ".csv", val_set.X_train, delimiter=",", fmt="%s",
           header="radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst", comments="")
		np.savetxt("data_validation_test_" + str(3) + "_" + str(i) + "_" + ".csv", val_set.X_test, delimiter=",", fmt="%s",
           header="radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst", comments="")
        
		np.savetxt("target_validation_train_"+ str(3) + "_" + str(i) + "_" + ".csv", val_set.y_train, delimiter=",", fmt="%s",
           header="diagnosis", comments="")
		np.savetxt("target_validation_test_" + str(3) + "_" + str(i) + "_" + ".csv", val_set.y_test, delimiter=",", fmt="%s",
           header="diagnosis", comments="")
		i += 1