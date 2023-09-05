import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(x1, x2):
    """
    Entradas:
    - x1: Un punto en el espacio n-dimensional.
    - x2: Otro punto en el espacio n-dimensional.

    Salida:
    - La distancia euclidiana entre los dos puntos.
    """
    return np.sqrt(np.sum((x1 - x2)**2))

# Función de ajuste KNN (no hace nada en este caso, solo devuelve los datos de entrenamiento)
def knn_fit(X_train, y_train):
    """
    Entradas:
    - X_train: Conjunto de datos de entrenamiento (características).
    - y_train: Etiquetas correspondientes a los datos de entrenamiento.

    Salida:
    - X_train: Conjunto de datos de entrenamiento (características) no modificado.
    - y_train: Etiquetas correspondientes a los datos de entrenamiento no modificadas.
    """
    return X_train, y_train

# Función de predicción KNN
def knn_predict(X_train, y_train, X_test, k=3):
    """
    Entradas:
    - X_train: Conjunto de datos de entrenamiento (características).
    - y_train: Etiquetas correspondientes a los datos de entrenamiento.
    - X_test: Conjunto de datos de prueba (características) para los que se realizarán predicciones.
    - k: El número de vecinos más cercanos a considerar para la predicción (por defecto, 3).

    Salida:
    - predictions: Una lista de etiquetas predichas para los datos de prueba.
    """
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

# Entradas: No se requieren argumentos, ya que el conjunto de datos es predefinido.
# Salidas:
# - iris: Un objeto que contiene datos y etiquetas del conjunto de datos Iris.

X = iris.data
y = iris.target

# Entradas:
# - iris: Objeto que contiene datos y etiquetas del conjunto de datos Iris.
# Salidas:
# - X: Conjunto de datos (características).
# - y: Etiquetas correspondientes a los datos.

# División en conjuntos de entrenamiento, prueba y validación
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

# Entradas:
# - X: Conjunto de datos completo (características).
# Salidas:
# - indices: Índices aleatoriamente reordenados para dividir los datos.

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Entradas:
# - train_ratio: Proporción del conjunto de entrenamiento.
# - val_ratio: Proporción del conjunto de validación.
# - test_ratio: Proporción del conjunto de prueba.
# Salidas:
# - train_split: Índice de división para el conjunto de entrenamiento.
# - val_split: Índice de división para el conjunto de validación.
# - X_train: Conjunto de datos de entrenamiento (características).
# - X_val: Conjunto de datos de validación (características).
# - X_test: Conjunto de datos de prueba (características).
# - y_train: Etiquetas correspondientes al conjunto de entrenamiento.
# - y_val: Etiquetas correspondientes al conjunto de validación.
# - y_test: Etiquetas correspondientes al conjunto de prueba.

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

# Entradas:
# - X_train: Conjunto de datos de entrenamiento (características).
# - y_train: Etiquetas correspondientes al conjunto de entrenamiento.
# Salidas:
# - X_train: Conjunto de datos de entrenamiento (características) no modificado.
# - y_train: Etiquetas correspondientes al conjunto de entrenamiento no modificadas.

# Realiza la predicción KNN en el conjunto de prueba
predictions = knn_predict(X_train, y_train, X_test, k=3)

# Entradas:
# - X_train: Conjunto de datos de entrenamiento (características).
# - y_train: Etiquetas correspondientes al conjunto de entrenamiento.
# - X_test: Conjunto de datos de prueba (características) para los que se realizarán predicciones.
# - k: El número de vecinos más cercanos a considerar para la predicción (por defecto, 3).
# Salidas:
# - predictions: Una lista de etiquetas predichas para los datos de prueba.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Métricas de evaluación

# Entradas:
# - y_test: Etiquetas verdaderas del conjunto de prueba.
# - predictions: Etiquetas predichas para el conjunto de prueba.
# Salidas:
# - accuracy: Exactitud (accuracy) de las predicciones.
# - precision: Precisión de las predicciones (weighted, promediada por clase).
# - recall: Recall de las predicciones (weighted, promediado por clase).
# - f1: Puntuación F1 de las predicciones (weighted, promediada por clase).
# - conf_matrix: Matriz de confusión de las predicciones.

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

# Ahora, agregar el código para probar diferentes valores de k, con esto demostrar que generaliza
k_values = [1, 3, 5, 7, 9]

for k in k_values:
    print(f"\nEvaluación para k = {k}:")

    # Realiza la predicción KNN en el conjunto de prueba con el valor de k actual
    predictions = knn_predict(X_train, y_train, X_test, k=k)

    # Calcular y mostrar métricas de evaluación para el valor de k actual
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


# Código para realizar predicciones en datos de ejemplo y en nuevos datos ingresados por el usuario

# Entradas:
# - sample_data: Conjunto de datos de ejemplo para los que se realizarán predicciones.
# Salidas:
# - sample_predictions: Una lista de etiquetas predichas para los datos de ejemplo.

# Usaremos los datos que agregue el usuario para probar el modelo

# Entradas:
# - num_samples: Número de muestras que el usuario ingresará.
# Salidas:
# - new_samples: Conjunto de datos ingresado por el usuario.
# - new_predictions: Una lista de etiquetas predichas para los datos ingresados por el usuario.

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
