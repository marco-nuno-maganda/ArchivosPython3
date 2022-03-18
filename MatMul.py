"""
Programa para estimar el algoritmo de Gradiente Descendiente en una red Neuronal
Basado en el Blog: https://mmuratarat.github.io/2020-01-09/backpropagation

Arquitectura de la Red: 5 entrada, 3 ocultas, 1 de salida

"""

import math

Alfa=0.5

# 5 muestras en el dataset (filas)
# 3 atributos (2 numericos, 1 categorico con 3 posibles valores = 1 hot encoding)
X = [[0.5, 0.1, 1, 0, 0],
    [0.3, 0.2, 0, 1, 0],
    [0.7, 0.9, 0, 0, 1],
    [0.8, 0.1, 1, 0, 0]]

# Objetivos de cada muestra (4 filas x 1 columna)
Target=[[0.1],
[0.6],
[0.4],
[0.1]]

# Pesos Entre la Capa de Entrada y la Capa Oculta
# 5 Entras (filas)
# 3 neuronas ocultas (columas)
thethas_IH = [[0.19, 0.55, 0.76],
        [0.33, 0.16, 0.97],
        [0.4, 0.35, 0.7],
        [0.51, 0.85, 0.85],
        [0.54, 0.49, 0.51]]
       
# Biases de la Capa Entrada-Oculta = 1 para cada Neurona Oculta
Biases_H=[[0.1],
[0.1],
[0.1]]
       
# Pesos entre la Capa Oculta-Salida (3 filas x 1 columa)
thethas_HO=[[0.1],
[0.03],
[-0.17]]

# Biases de la Capa Entrada-Oculta = 1 para cada Neurona de Salida
B2=0.1


XR=[]
for N in range(3):
    resultado = []
    for elemento, indice in zip(X[0], thethas_IH):
        resultado.append(elemento * indice[N])

    R=sum(resultado) + 1*Biases_H[N][0]
    print(R)
    Sig = 1 / (1 + math.pow(math.e, R*-1))
    print (Sig)
    XR.append(Sig)

print (XR)
Sal_NeuronasOcultas=[]

for N in range(1):
    resultado = []
    for elemento, indice in zip(XR, thethas_HO):
        resultado.append(elemento * indice[N])
#    R=sum(resultado) + 1*0.1
    R=sum(resultado) + 1*B2
    print(R)
    Sal_NeuronasOcultas.append(R)
   

print (Sal_NeuronasOcultas)    
error_total = (Target[0][0] - Sal_NeuronasOcultas[0])**2
print (error_total)
derivada_error_total = -2*(Target[0][0] - Sal_NeuronasOcultas[0])
print (derivada_error_total)

B2 = B2 - (Alfa * derivada_error_total)
print (B2)


# Actualizar los Biases
print ("Nuevos Biases")
for N in range(3):
    #print (N)
    #print (Alfa)
    #print (derivada_error_total)
    #print (Biases_H[N][0])
    #print (1-XR[N])
    #print (XR[N])
    Biases_H[N][0] = Biases_H[N][0] - Alfa * (derivada_error_total*thethas_HO[N][0]*XR[N]*(1-XR[N])*1)
    print (Biases_H[N][0])
    



print ("Nuevos Tethas IO")
for NumNeuronaOculta in range(3):
    print ("Neurona Oculta # "+str(NumNeuronaOculta))
    for NumEntrada in range(5):
        print  ("NumEntrada"+str(NumEntrada))
        print (thethas_IH[NumEntrada][NumNeuronaOculta])
        print (Alfa)
        print (derivada_error_total)  
        print (thethas_HO[NumNeuronaOculta][0])
        print (XR[NumNeuronaOculta])
        print (1-XR[NumNeuronaOculta])        
        thethas_IH[NumEntrada][NumNeuronaOculta] = thethas_IH[NumEntrada][NumNeuronaOculta] - Alfa * (derivada_error_total*thethas_HO[NumNeuronaOculta][0]*XR[NumNeuronaOculta]*(1-XR[NumNeuronaOculta])*X[NumNeuronaOculta][NumEntrada])
        print (thethas_IH[NumEntrada][NumNeuronaOculta])

print ("Nuevos Tethas HO")
#  Derivada por la salida de cada neurona oculta (dentro del parenteris)
# Ese producto multiplicarlo por Alfa y restado del Pesos
thethas_HO[0][0] = thethas_HO[0][0] - Alfa * (XR[0]*derivada_error_total)
print(thethas_HO[0][0])

thethas_HO[1][0] = thethas_HO[1][0] - Alfa * (XR[1]*derivada_error_total)
print(thethas_HO[1][0])

thethas_HO[2][0] = thethas_HO[2][0] - Alfa * (XR[2]*derivada_error_total)
print(thethas_HO[2][0])

exit()        

