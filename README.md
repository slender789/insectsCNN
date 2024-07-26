# insectsCNN

Este proyecto se centra en la detección de insectos utilizando una red neuronal convolucional (CNN). A continuación se describen los archivos principales del proyecto:

- **Archivos de imagen**: Los archivos que comienzan con `all_*` contienen las imágenes utilizadas para entrenar el modelo de inteligencia artificial.

- **`use_weights.py`**: Utiliza los pesos del modelo entrenado para predecir la categoría de una imagen dada.

- **`preprocess_raw_image.py`**: Realiza el preprocesamiento de la imagen inicial, preparando los datos para la segmentación.

## Datos

1. **Tiempos de ejecución**: El promedio del tiempo de segmentación por imagen es de 0.81 segundos, mientras que el promedio de realizar la predicción de cada uno de los segmentos por imagen es de 10.29 segundos.

2. **Modelos**: Los modelos entrenados hasta el momento son CNN, ANN, KNN, NV y Kernel SVM.
| Salt & Pepper   | Row 1, Col 2    | Row 1, Col 3    |

## Precisión por Modelo

1. **CNN**: Accuracy: 0.9861.

2. **ANN**: Accuracy: 0.9722.

3. **KNN**: Accuracy: 0.9166.

4. **NV**: Accuracy: 0.9166.

5. **Kernel SVM**: Accuracy: 0.9722.

## Precisión CNN según preprocesamiento utilizado en el entrenamiento

| Kernel          | Accuracy  | Loss            |
| --------------- | ----------| --------------- |
| Without         | 96%       | 13.14%          |
| Salt & Pepper   | 100%      | 2.7%            |
| Poisson         | 100%      | 4.43%           |
| Rotation        | 93.04%    | 24.15%          |
| All 3           | 98.61%    | 15.91%          |
