import pandas as pd
import numpy as np # import NumPy with alias np

series1 = pd.Series([10,20,30])
print (series1)

series2 = pd.Series(["Karen","Jose Carlos","Jacobo"],index=[8,7,6])
print (series2)

series3 = pd.Series([10,15,20],index=["Enero","Febrero","Marzo"])
print (series3)

series4 = pd.Series([[5,5,7],[8,7,3],[9,2,6]],index=["Enero","Febrero","Marzo"])
print (series4)

array1 = np.array([1,2,3,4])
series5 = pd.Series(array1)
print(series5)

series6 = pd.Series(array1, index = ["Jan",
"Feb", "Mar","Abr"])
print(series6)

dict1 = {'Tamaulipas': 'Ciudad Victoria', 'Nuevo Leon':'Monterrey', 'Queretaro': 'Queretaro', 'Nayarit': 'Tepic'}
series7 = pd.Series(dict1)
print(series7)


print(series4['Enero'])
print(series7['Tamaulipas'])
print(series7[0])
#series7.index=['Texas','Nue Mexico','Estado de Mexico','Jalisco']
#print(series7['Jalisco'])
#print(series7[0])

#print(series7[len(series7)-1])
print(series7[-1])

dFrameEmt = pd.DataFrame()
print(dFrameEmt)

array1= np.array([10,20,30])
array2= np.array([100,200,300])
array3= np.array([-10,-20,-30, -40])

dFrame4 = pd.DataFrame(array1)
print (dFrame4)

dFrame5 = pd.DataFrame([array1,array2,array3],columns=['A','B','C','D'])
print (dFrame5)

dFrame6 = pd.DataFrame([array1,array2,array3],index=["Juan","Beto","Luis"], columns=['A','B','C','D'])
print (dFrame6)








exit()


























series2 = pd.Series(["Josue","Mariana","Francisco"], index =[7,5,2])
print(series2)

series3 = pd.Series([10,15,20], index =["Enero","Febrero","Marzo"])
print(series3)

array1 = np.array([1,2,3,4])
series3 = pd.Series(array1)
print(series3)

dFrameEmt = pd.DataFrame()
print(dFrameEmt)

array1= np.array([10,20,30])
array2= np.array([100,200,300])
array3= np.array([-10,-20,-30, -40])
dFrame4 = pd.DataFrame(array1)
print(dFrame4)

dFrame5 = pd.DataFrame([array1, array3, array2], columns=[ 'A', 'B', 'C', 'D'])
print (dFrame5)

#listDict = [{'a':10, 'b':20}, {'a':5,'b':10, 'c':20}]
listDict = [{'a':10, 'b':20, 'd':500}, {'a':5,'b':10, 'c':20}]
dFrameListDict = pd.DataFrame(listDict)
print (dFrameListDict)

seriesA = pd.Series([1,2,3,4,5],
index = ['a', 'b', 'c', 'd', 'e'])

seriesB = pd.Series ([1000,2000,-1000,-5000,1000],		
index = ['a', 'b', 'c', 'd', 'e'])

seriesC = pd.Series([10,20,-10,-50,100],		
index = ['z', 'y', 'a', 'c', 'e'])

dFrame6 = pd.DataFrame(seriesA)
print(dFrame6)

dFrame7 = pd.DataFrame([seriesA, seriesB])
print(dFrame7)





