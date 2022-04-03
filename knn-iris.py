import pandas as pd
import seaborn as sn
from shlex import split
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split #Train-test split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


iris = load_iris()

iris.feature_names
print ("\nArray: " + str(iris.feature_names))

iris.target_names
print ("\nArray: " + str(iris.target_names))

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
print ("\nArray: " + str(df.head()))

df['target'] = iris.target
df.head()
print ("\n" + str(df.head()))

df[df.target==1].head()
print ("\n" + str(df[df.target==1].head()))

df[df.target==2].head()
print ("\n" + str(df[df.target==1].head()))

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()
print ("\n" + str(df.head()))

df[45:55]
print ("\n" + str(df))

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]
#Longitud de Sepalo vs Ancho de Sepalo (Setosa vs Versicolor)
plt.xlabel ('Sepal Length')
plt.ylabel ('Sepal Width')
plt.scatter (df0['sepal length (cm)'], df0['sepal width (cm)'], color = "green", marker = '+')
plt.scatter (df1['sepal length (cm)'], df1['sepal width (cm)'], color = "blue", marker = '.')

plt.show()

#Longitud de Pepalo vs Ancho de Pepalo (Setosa vs Versicolor)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')

plt.show()

X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
len(X_train)
len(X_test)

print ("\nEntrenamiento: " + str(len(X_train))) #Se convierte a string el len
print ("\nPrueba: " + str(len(X_test))) #Se convierte a string el len

knn = KNeighborsClassifier(n_neighbors = 10) #Creación del clasificador KNN a 10 vecinos
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print ("\nPuntuación: " + str(knn.score(X_test, y_test)))

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print ("\nMatriz de confusión: \n\n" + str(cm))

plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()

print(classification_report(y_test, y_pred))