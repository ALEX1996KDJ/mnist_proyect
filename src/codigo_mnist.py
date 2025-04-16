# 📦 Importación de librerías
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# 📂 Crear carpetas para guardar resultados
os.makedirs("modelo_guardado", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

# 📥 Carga de datos MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 🔧 Normalización
x_train = x_train / 255.0
x_test = x_test / 255.0

# 🧠 Definición del modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 clases porque MNIST va del 0 al 9
])

# ⚙️ Compilación del modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # porque las etiquetas son enteros (no one-hot)
    metrics=['accuracy']
)

# 🏋️ Entrenamiento del modelo
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=1
)

# 💾 Guardar el modelo entrenado
model.save("modelo_guardado/mi_modelo.h5")

# 📈 Gráficas de entrenamiento
plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Guardar imagen
plt.savefig("graficas/entrenamiento.png")
plt.show()
