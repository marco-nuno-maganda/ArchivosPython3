import numpy as np
from sklearn import preprocessing
from sklearn import datasets

# Sample input labels
#input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white', 'cyan', 'magenta']

iris = datasets.load_iris()
input_labels = iris.target_names

# Create label encoder and fit the labels
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

clases = iris.target
L=encoder.inverse_transform(iris.target)
print(clases)
print(L)

encoded_values = encoder.transform(L)
print("Encoded values =", list(encoded_values))

encoded_values = encoder.transform(['versicolor'])
print("Encoded values =", list(encoded_values))

exit()

print (input_labels)



# Print the mapping
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Encode a set of labels using the encoder
#test_labels = ['green', 'red', 'black']
test_labels = ['setosa', 'versicolor', 'virginica', 'virginica','setosa', 'versicolor']
#test_labels = input_labels

encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

exit()

ListaX=[6,0,1,2,3,4,5]
L=encoder.inverse_transform(ListaX)
print(ListaX)
print(L)

exit()

L=encoder.inverse_transform(iris.target)
print(L)
exit()

