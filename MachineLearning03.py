from sklearn import datasets
from sklearn import preprocessing
print(datasets)
iris = datasets.load_iris()

print(iris.data)
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(iris.data)
#print(data_scaled_minmax)

# Normalize data
data_normalized_l1 = preprocessing.normalize(iris.data, norm='l1')
data_normalized_l2 = preprocessing.normalize(iris.data, norm='l2')
print("\nL1 normalized data:\n")
print(data_normalized_l1)
print("\nL2 normalized data:\n")
print(data_normalized_l2)
