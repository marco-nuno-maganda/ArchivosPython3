"""
Clasificadores obtenidos del sig. enlace:
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


"""

# evaluate a logistic regression model using k-fold cross-validation

from numpy import mean
from numpy import std
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# create dataset
#X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
iris = datasets.load_iris()
X=iris.data
y=iris.target
# Punto importante: Normalizar los datos, porque algunos algoritmos marcaran error
X = preprocessing.scale(X)

# prepare the cross-validation procedure
#cv = KFold(n_splits=10, random_state=1, shuffle=True)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model

#model = LogisticRegression()  #VIEJA (Logistics Regresion)
model = LogisticRegression(solver='liblinear', C=1) #  NUEVA (Acutal)
model = GaussianNB()
model = OneVsOneClassifier(LinearSVC(random_state=0))
model = DecisionTreeClassifier(max_depth=5)
model = KNeighborsClassifier(3)

model.fit(X, y) # Entrenar ...
y_pred = model.predict(X) # Probar con los datos de entrada (Entramiento)
print (y)
print (y_pred)
print (abs(y-y_pred))
print(sum(abs(y-y_pred)))

# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
