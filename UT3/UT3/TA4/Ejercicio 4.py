import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

input_file = "cardiac-training.csv"
df = pd.read_csv(input_file, header=0)
print(df.values)

X = df.loc[:, df.columns != '2do_Ataque_Corazon']
y = df['2do_Ataque_Corazon'].values
print(X)
print(y)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.30, random_state=
0, shuffle=True)

# Crear el modelo de regresión logística con regularización L2 y solver 'liblinear'
lr = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, random_state=0, max_iter=100)

# Entrenar el modelo con los datos de entrenamiento
lr.fit(train_X, train_y)

# Predecir las clases en el conjunto de prueba
predictions = lr.predict(test_X)

# Imprimir reporte de clasificación
print(classification_report(test_y, predictions))

#Imprimir el reporte de confusión
print(confusion_matrix(test_y, predictions))


