from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from  sklearn.metrics import accuracy_score
from  sklearn.metrics import precision_score
from  sklearn.metrics import recall_score
from  sklearn.metrics import f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def train(x_train, y_train):
    classifiers_name = ['MLP']
    classifiers = [MLPClassifier()]    
    best_score = 0
    for clf_name, clf in zip(classifiers_name, classifiers):
        if clf_name == 'MLP':
            parameters =  [{'clf__activation': ['logistic', 'relu'], 'clf__alpha': [.0001,.001], 'clf__learning_rate': ['constant', 'adaptive'], 
                   'clf__max_iter': [1000, 2000], 'clf__hidden_layer_sizes':[(1000,70),(1000,20)]}]

        pipeline = Pipeline([
                            ('clf', clf)
                            ])
        #~ print (pipeline)
        #~ cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=2)
        cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1, cv=5)
        cv.fit(x_train, y_train)
        print ("Classifier: " + clf_name)
        cv_best_score = cv.best_score_
        print("\nBest score: {:0.3f}".format(cv_best_score))
        print("\nBest parameters set:")
        best_parameters = cv.best_estimator_.get_params()
        for dic in parameters:
            for param_name in sorted(dic.keys()):
                print("\t{}: {}".format(param_name, best_parameters[param_name]))
            
        if     cv_best_score > best_score:
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
    
    print("El accuracy es:",accuracy_score(y_test, y_predictions))
    
    
    print("El precision es:",precision_score(y_test, y_predictions, average='weighted'))
    
   
    print("El recall es:",recall_score(y_test, y_predictions, average='weighted'))

    print("El F-Measure es:",f1_score(y_test, y_predictions, average='weighted'))
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_predictions)
    plt.show()

    for i in range(len(y_test)):
        if (y_test[i]!=y_predictions[i]) :
            img = X_test[i].reshape((28,28))
            plt.imshow(img, cmap="Greys")
            plt.title('Real: ' + str(y_test[i]) + ' Predicted: ' + str(y_predictions[i]), fontsize = 20);
            plt.show()
            
#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
print ('Cargando datos...')
df = pd.read_csv('mnist_train.csv', sep=',', engine='python')
X = df.drop(['label'],axis=1).values   
y = df['label'].values

df_test = pd.read_csv('mnist_test.csv', sep=',', engine='python')
X_test = df_test.drop(['label'],axis=1).values   
y_test = df_test['label'].values

model = MLPClassifier(activation="logistic", max_iter=500, hidden_layer_sizes=(24*24,24), alpha=0.0001)

model = model.fit(X,y)
test(model, X_test, y_test)