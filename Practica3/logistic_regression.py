import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_set = []

for i in range(1,4):
    X_train = pd.read_csv('data_validation_train_3_'+str(i)+'_.csv', sep=',', engine='python')
    X_test = pd.read_csv('data_validation_test_3_'+str(i)+'_.csv', sep=',', engine='python')
    y_train = pd.read_csv('target_validation_train_3_'+str(i)+'_.csv', sep=',', engine='python')
    y_test = pd.read_csv('target_validation_test_3_'+str(i)+'_.csv', sep=',', engine='python')
    y_train = y_train.values.ravel()
    clf = LogisticRegression(solver="lbfgs" ,penalty="none", max_iter=100000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_set.append(accuracy_score(y_test, y_pred))

print(accuracy_set)
print(np.mean(accuracy_set))
#Comparando con el conjunto de pruebas
X_test = pd.read_csv('data_test_3folds.csv', sep=',', engine='python')
y_test = pd.read_csv('target_test_3folds.csv', sep=',', engine='python')
y_pred_test = clf.predict(X_test)
print(accuracy_score(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))