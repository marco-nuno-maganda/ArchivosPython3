import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from utilities import visualize_classifier
from sklearn import datasets

# Define sample input data
#X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
#y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


X1 = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9], [2,7]])
y1 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9], [2,7]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

X2 = np.array([[1, 1], [1, 3], [1, 5], [2, 1], [2, 3], [2, 5], [6, 1], [6, 3], [6, 5], [7, 1], [7, 3], [7, 5], [10,1], [10,3], [10,5]])
y2 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 , 1])

X3 = np.array([[1, 1], [1, 3], [1, 5], [2, 1], [2, 3], [2, 5], [6, 1], [6, 3], [6, 5], [7, 1], [7, 3], [7, 5], [10,1], [10,3], [10,5]])
y3 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])

X4 = np.array([[1, 1], [1, 3], [1, 5], [2, 1], [2, 3], [2, 5], [6, 1], [6, 3], [6, 5], [7, 1], [7, 3], [7, 5], [10,1], [10,3], [10,5], [1,7],[2,7],[6,7],[7,7],[10,7]])

#y4 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,0])
 y4 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,1])

# Create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier2 = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier3 = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier4 = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier5 = linear_model.LogisticRegression(solver='liblinear', C=1)

# Train the classifier
classifier.fit(X, y)
classifier2.fit(X1, y1)
classifier3.fit(X2, y2)
classifier4.fit(X3, y3)
classifier5.fit(X4, y4)

# Visualize the performance of the classifier
#visualize_classifier(classifier, X, y)
#visualize_classifier(classifier2, X1, y1)
#visualize_classifier(classifier3, X2, y2)
#visualize_classifier(classifier4, X3, y3)
visualize_classifier(classifier5, X4, y4)

plt.show()


exit()

iris = datasets.load_iris()
X = iris.data[:,0:2]
#print (X)

print (iris.target_names)



X1 = iris.data[:,1:3]
#print (X1)

y= iris.target

# Create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier2 = linear_model.LogisticRegression(solver='liblinear', C=1)

# Train the classifier
classifier.fit(X, y)
classifier2.fit(X1, y)

# Visualize the performance of the classifier
visualize_classifier(classifier, X, y)
visualize_classifier(classifier2, X1, y)
plt.show()


exit()


