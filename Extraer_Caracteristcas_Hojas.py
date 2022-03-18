from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.color import rgb2gray
from skimage.io import imshow, imread, imsave
import pandas as pd
from skimage.data import binary_blobs
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



#Paso 0: Checar versiones:
import sys; print(sys.version)
import platform; print(platform.platform())
import skimage; print("scikit-image version: {}".format(skimage.__version__))
import numpy; print("numpy version: {}".format(numpy.__version__))
import pandas; print("pandas version: {}".format(pandas.__version__))


#Archivo_Imagen="Hoja7.png"
#Archivo_Imagen="Hoja6.png"
#Archivo_Imagen="Hoja6_M.png"
#Archivo_Imagen="Hoja5.png"
#Archivo_Imagen="Hoja4.png"
#Archivo_Imagen="Hoja4_M.png"
#Archivo_Imagen="Hoja3.png"
#Archivo_Imagen="Hoja3_M.png"
#Archivo_Imagen="Hoja2.png"
#Archivo_Imagen="Hoja2_M.png"
#Archivo_Imagen="Hoja1.png"
Clases=7
df = pd.DataFrame()

for Indice in range (1,Clases+1):

    Archivo_Imagen="Clase"+str(Indice)+".png"

    #image = data.coins()[50:-50, 50:-50]
    #image = rgb2gray(imread("Letras2.png"))
    #image = rgb2gray(imread("Hoja1.png"))
    #image = rgb2gray(imread("Hoja2.png"))
    #image = rgb2gray(imread("Hoja3.png"))
    #image = rgb2gray(imread("Hoja4.png"))
    #image = rgb2gray(imread("Hoja5.png"))
    color_image = imread(Archivo_Imagen)
    image = rgb2gray(color_image)

    # apply threshold
    thresh = threshold_otsu(image)
    #bw = closing(image > thresh, square(3))
    bw = closing(image > thresh)

    bw = numpy.invert(bw)


    plt.imshow(bw)

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    #table = pd.DataFrame(regionprops_table(label_image, image
    #	['convex_area', 'area', 'eccentricity',
    #	'extent', 'inertia_tensor','major_axis_length', 'minor_axis_length',
    #	'perimeter', 'solidity', 'image',
    #	'orientation', 'moments_central',
    #	'moments_hu', 'euler_number',
    #	'equivalent_diameter',
    #	'mean_intensity', 'bbox']))

    table = pd.DataFrame(regionprops_table(label_image, intensity_image=image,
	    properties=['convex_area', 'area', 'eccentricity','extent', 'inertia_tensor','major_axis_length', 'minor_axis_length','perimeter']))
    table['perimeter_area_ratio'] = table['perimeter']/table['area']


    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    real_images = []
    std = []
    mean = []
    percent25 = []
    percent75 = []


    for prop in regionprops(label_image):
	    # take regions with large enough areas
	    #if prop.area >= 800:
	    # draw rectangle around segmented coins
	    minr, minc, maxr, maxc = prop.bbox
	    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
	    ax.add_patch(rect)

	    min_row, min_col, max_row, max_col = prop.bbox
	    #img = image[min_row:max_row,min_col:max_col]
	    img = color_image[min_row:max_row,min_col:max_col]
	    #real_images += [img]
	    mean += [np.mean(img)]
	    std += [np.std(img)]
	    percent25 += [np.percentile(img, 25)] 
	    percent75 += [np.percentile(img, 75)]


    #table['real_images'] = real_images
    table['mean_intensity'] = mean
    table['std_intensity'] = std
    table['25th Percentile'] = mean
    table['75th Percentile'] = std
    table['iqr'] = table['75th Percentile'] - table['25th Percentile']

    #table['label'] = Archivo_Imagen
    table['class'] = Indice
    df = pd.concat([df, table], axis=0)

#print(df)
print (df.columns)
ListaFeatures=list(df.columns.values);
print(ListaFeatures)
ListaFeatures.pop()
print(ListaFeatures)

training_features = df[ListaFeatures]
print (training_features)

outcome_name = ['class']
outcome_labels = df[outcome_name]
print(outcome_labels)

classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier.fit(training_features, np.array(outcome_labels['class']))

#pred_labels = classifier.predict(training_features)
pred_labels = classifier.predict(training_features)
actual_labels = np.array(outcome_labels['class'])

print (pred_labels)
print (actual_labels)

print('Accuracy:', float(accuracy_score(actual_labels,
                pred_labels))*100, '%')               
                
                
PorcentajeTestSize=0.3                

for Experimento in range(1,8):
    X_train, X_test, y_train, y_test = train_test_split(training_features, np.array(outcome_labels['class']), test_size=PorcentajeTestSize, random_state=5)

    classifier_new = linear_model.LogisticRegression(solver='liblinear', C=1)
    classifier_new.fit(X_train, y_train)
    y_test_pred = classifier_new.predict(X_test)

    # compute accuracy of the classifier
    accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
    #print("Accuracy of Split ",str(PorcentajeTestSize)," =", round(accuracy, 2), "%")
    print("Accuracy of Split ","{:2.2f}".format(PorcentajeTestSize)," =", "{:2.2f}".format(round(accuracy, 2)), "%")
    PorcentajeTestSize+=0.1

df.to_csv('PinceDataset.csv')




#print (mean)
#print (std)
#print (percent25)
#print (percent75)

# Mostrar cada imagen...
#Contador=1
#for i in real_images:
#    plt.imsave(''+Archivo_Imagen+'_Img'+str(Contador)+'.png', i, cmap = plt.cm.gray)
#    print ("imagen ",Contador,i.shape)
#    Contador+=1


ax.set_axis_off()
plt.tight_layout()
#plt.show()


exit()

#blobs = binary_blobs()

#image = rgb2gray(imread("Hoja1.png"))
image = rgb2gray(imread("Hoja2.png"))
thresh = threshold_otsu(image)
blobs = closing(image > thresh)
#imshow(blobs)
plt.imshow(blobs)

blobs_table = regionprops_table(label(blobs), intensity_image=image, properties=['centroid', 'local_centroid'])
print("regionprops table result")
print(blobs_table)

blobs_table = regionprops_table(label(blobs), intensity_image=image, properties=['convex_area', 'area', 'eccentricity',
    'extent', 'inertia_tensor','major_axis_length', 'minor_axis_length',
    'perimeter', 'solidity', 'image',
    'orientation', 'moments_central',
    'moments_hu', 'euler_number',
    'equivalent_diameter',
    'mean_intensity',
    'bbox'])
print("regionprops table result")
print(blobs_table)


exit()

