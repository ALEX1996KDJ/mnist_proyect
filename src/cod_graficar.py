
#graficas
import matplotlib.pyplot as plt
import os
import pickle

# Crear carpeta si no existe

os.makedirs("graficas", exist_ok=True)

# Cargar historial
with open("modelo_guardado/history.pkl", "rb") as f:
    history = pickle.load(f)

# Crear gráficas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Precisión entrenamiento')
plt.plot(history['val_accuracy'], label='Precisión validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Pérdida entrenamiento')
plt.plot(history['val_loss'], label='Pérdida validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.savefig("graficas/entrenamiento.png")
plt.show()
