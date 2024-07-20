# insectsCNN

Este proyecto se centra en la detección de insectos utilizando una red neuronal convolucional (CNN). A continuación se describen los archivos principales del proyecto:

- **Archivos de imagen**: Los archivos que comienzan con `all_*` contienen las imágenes utilizadas para entrenar el modelo de inteligencia artificial.

- **`use_weights.py`**: Utiliza los pesos del modelo entrenado para predecir la categoría de una imagen dada.

- **`preprocess_raw_image.py`**: Realiza el preprocesamiento de la imagen inicial, preparando los datos para la segmentación.

## Datos

1. **Tiempos de ejecución**: El promedio del tiempo de segmentación por imagen es de 0.81 segundos, mientras que el promedio de realizar la predicción de cada uno de los segmentos por imagen es de 10.29 segundos.

2. **Modelos**: Los modelos entrenados hasta el momento son CNN, ANN, KNN, NV y Kernel SVM.

## Precisión por Modelo

1. **CNN**: Accuracy: 0.9685.

2. **ANN**: Accuracy: 0.7826.

3. **KNN**: Accuracy: 0.8695.

4. **NV**: Accuracy: 0.6956.

5. **Kernel SVM**: Accuracy: 0.9130.
