import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Función de ajuste KNN (no hace nada en este caso, solo devuelve los datos de entrenamiento)
def knn_fit(X_train, y_train):
    return X_train, y_train

# Función de predicción KNN
def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for i, x in enumerate(X_test):
        # Calcula las distancias entre el punto de prueba (x) y todos los puntos de entrenamiento
        distances = [euclidean_distance(x, x_train) for x_train in X_train]

        # Encuentra los índices de las k instancias más cercanas
        k_indices = np.argsort(distances)[:k]

        # Obtiene las etiquetas correspondientes a las instancias más cercanas
        k_nearest_labels = [y_train[i] for i in k_indices]

        # Encuentra la etiqueta más común entre las vecinas cercanas
        most_common = Counter(k_nearest_labels).most_common(1)

        # Agrega la etiqueta más común a las predicciones
        predictions.append(most_common[0][0])

        # Imprime el resultado de la predicción y la etiqueta real para esta instancia
        print(f"Instance {i+1} - Predicted: {most_common[0][0]}, Actual: {y_test[i]}")

    return predictions

# Cargar el conjunto de datos (por ejemplo, iris)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# División en conjuntos de entrenamiento, prueba y validación
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

# Proporciones para la división
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Calcular los índices para cada conjunto
train_split = int(train_ratio * len(X))
val_split = int((train_ratio + val_ratio) * len(X))

X_train = X[indices[:train_split]]
X_val = X[indices[train_split:val_split]]
X_test = X[indices[val_split:]]

y_train = y[indices[:train_split]]
y_val = y[indices[train_split:val_split]]
y_test = y[indices[val_split:]]

# Realiza el ajuste KNN (que no hace nada más que devolver los datos de entrenamiento)
X_train, y_train = knn_fit(X_train, y_train)

# Realiza la predicción KNN en el conjunto de prueba
predictions = knn_predict(X_train, y_train, X_test, k=3)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ...

# Realiza la predicción KNN en el conjunto de prueba
predictions = knn_predict(X_train, y_train, X_test, k=3)

# Calcular y mostrar métricas de evaluación
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
conf_matrix = confusion_matrix(y_test, predictions)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

# ... Código para realizar predicciones en datos de ejemplo y en nuevos datos ingresados por el usuario ...
print("Aquí veremos un ejemplo de valores fijos y su resultado:")
sample_data = np.array([[5.1, 3.5, 1.4, 0.2],
                        [6.7, 3.1, 5.6, 2.4]])

sample_predictions = knn_predict(X_train, y_train, sample_data, k=3)

for i, prediction in enumerate(sample_predictions):
    print(f"Sample {i+1} predicted class:", prediction)
# Usaremos los datos que agregue el usuario para probar el modelo
num_samples = int(input("Ingrese cuantas predicciones hará: "))
new_samples = []
for _ in range(num_samples):
    sample = []
    for i in range(X.shape[1]):
        value = float(input(f"Entre el valor del parámetro {i+1}: "))
        sample.append(value)
    new_samples.append(sample)

new_samples = np.array(new_samples)

new_predictions = knn_predict(X_train, y_train, new_samples, k=3)

for i, prediction in enumerate(new_predictions):
    print(f"Sample {i+1} predicted class:", prediction)

