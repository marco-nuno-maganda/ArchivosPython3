import numpy as np
from sklearn import preprocessing
from sklearn import datasets

input_data = np.array([[5.1, -2.9, 3.3],
[-1.2, 7.8, -6.1],
[3.9, 0.4, 2.1],
[7.3, -9.9, -4.5]])

iris = datasets.load_iris()
input_data = iris.data
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

#print (input_data)
# Normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
print("Mean =", data_normalized_l1.mean(axis=0))
print("Std deviation =", data_normalized_l1.std(axis=0))
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("Mean =", data_normalized_l2.mean(axis=0))
print("Std deviation =", data_normalized_l2.std(axis=0))

#print("\nL1 normalized data:\n", data_normalized_l1)
#print("\nL2 normalized data:\n", data_normalized_l2)

exit()

print("Mean =", input_data.mean(axis=0))
print("Mean =", data_normalized_l1.mean(axis=0))
print("Mean =", data_normalized_l2.mean(axis=0))


iris = datasets.load_iris()
input_data = iris.data
data_scaled = preprocessing.scale(input_data)
#print ("\Original data:\n",input_data)
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
data_scaled2 = preprocessing.scale(data_scaled_minmax)
#print("\nMin max scaled data:\n", data_scaled_minmax)
print("Mean =", data_scaled2.mean(axis=0))
print("Std deviation =", data_scaled2.std(axis=0))

exit()


# Estima la Media y la Desviacion Estandard
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Remove mean
data_scaled = preprocessing.scale(input_data)
print ("Datos escalados",data_scaled)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

exit()



data_binarized = preprocessing.Binarizer(threshold=1.9).transform(input_data)
print("\nBinarized data:\n", data_binarized)

exit()




#L1 = Least Absolute Deviations
#L2 = Least Squares

# Normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)

print("Mean =", input_data.mean(axis=0))
print("Mean =", data_normalized_l1.mean(axis=0))
print("Mean =", data_normalized_l2.mean(axis=0))

exit()


# Min max scaling
#data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
print (data_scaler_minmax)
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)


exit()



# Estima la Media y la Desviacion Estandard
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Remove mean
data_scaled = preprocessing.scale(input_data)
print ("Datos escalados",data_scaled)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))


