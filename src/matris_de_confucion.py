#matris de confucion
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Cargar modelo
modelo = tf.keras.models.load_model("modelo_guardado/mi_modelo.h5")

# Cargar datos de prueba
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0  # Normalizar

# Obtener predicciones
y_pred_probs = modelo.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Crear matriz de confusión
matriz = confusion_matrix(y_test, y_pred)

# Visualización
plt.figure(figsize=(10, 8))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.title("📊 Matriz de Confusión - MNIST")

# Crear carpeta si no existe
os.makedirs("graficas", exist_ok=True)
plt.savefig("graficas/matriz_confusion.png")
plt.show()
