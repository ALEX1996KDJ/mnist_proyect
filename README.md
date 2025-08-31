Proyecto de Clasificación de Dígitos MNIST con Redes Neuronales
Este proyecto implementa una red neuronal artificial para clasificar dígitos manuscritos del conjunto de datos MNIST.

📋 Descripción
El proyecto incluye:

Entrenamiento de un modelo de red neuronal

Evaluación del rendimiento del modelo

Visualización de métricas de entrenamiento

Matriz de confusión para análisis detallado

🚀 Características
Arquitectura de Red Neuronal:

Capa Flatten para aplanar imágenes 28x28

3 capas Dense (128, 64, 64 neuronas) con activación ReLU

Dropout (25%) para prevenir overfitting

Capa final Softmax para clasificación de 10 clases

Preprocesamiento:

Normalización de píxeles (0-255 → 0-1)

División automática en entrenamiento y prueba

Métricas:

Precisión de entrenamiento y validación

Pérdida durante el entrenamiento

Matriz de confusión detallada

📊 Resultados Esperados
El modelo alcanza típicamente:

Precisión de entrenamiento: >95%

Precisión de validación: >90%

Buen rendimiento en la matriz de confusión

como ejecutar

pip install tensorflow matplotlib scikit-learn seaborn numpy