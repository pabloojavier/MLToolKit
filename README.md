# Machine Learning Tool Kit

Este repositorio contiene herramientas para realizar modelos de machine learning.

## Contenido

- [Instalación](#instalación)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Instalación

Para instalar las dependencias necesarias, ejecute:

```bash
pip install git+https://github.com/pabloojavier/MLToolKit
```

## Uso

### Análisis de variables

Para utilizar las herramientas se deben configurar unos parámetros, por ejemplo, para utilizar la librería dentro de databricks se debe configurar el parámetro `databricks` en `True`. Además, se debe especificar la ruta completa donde se creará el experimento de mlflow para descargar los archivos. A continuación se muestra un ejemplo de configuración en databricks:

```python
features = ['feature_1','feature_2','feature_3','feature_4']
parametros = {
  'xgboost_params' : {"random_state": 42, "max_depth": 2},
  'experiment_name' : "test",
  'nombre_reporte' : 'local_test',
  'databricks' : False,
  'target_name' : 'target',
  'periodo_id' : 'event_time',
  'cliente_id' : 'cliente_id',
  'features' : features,
  'threshold_low_variabilty' : 0.95,
  'fill_na' : -9e8, 
  'threshold_correlation' : 0.5,
  'correlation_metric' : 'aucpr', # Opciones: iv,ks,roc,aucpr
  'training_period' : pd.date_range('2022-01-01', '2022-08-01', freq='MS'),
  'verbose' : True
}


from MLToolKit.feature_analysis import FeatureAnalysis
fa = FeatureAnalysis(df,parametros)
fa.univariado()
fa.estabilidad()
fa.correlacion_por_metrica()
fa.consolidar_analisis()
fa.save_files()
```

### Parámetros

- xgboost_params: Diccionario con los parámetros del modelo xgboost para análisis univariado.
- experiment_name: En caso de utilizar mlflow, se debe especificar el nombre del experimento. __Recomiendo usar la ruta completa a la carpeta de trabajo__.
- nombre_reporte: Nombre de los reportes que se crearán.
- databricks: Booleano que indica si se está trabajando en databricks o no.
- target_name: Nombre de la marca de desempeño. En caso de no tener, se puede agregar una columna al dataframe con valores constantes.
- periodo_id: Nombre de la columna que contiene la fecha de los datos. Recomiendo que sea en formato string yyyy-mm-dd.
- cliente_id: Nombre de la columna que contiene el id del cliente.
- features: Lista con los nombres de las variables a analizar.
- threshold_low_variabilty: Umbral para eliminar variables con baja variabilidad.
- fill_na: Valor con el que se reemplazarán los valores nulos. Recomiendo que el dataframe __contenga nulos__, en los análisis correspondientes se reemplazarán por este valor.
- threshold_correlation: Umbral para eliminar variables con alta correlación familiar.
- correlation_metric: Métrica para escoger que variable seleccionar en caso de tener alta correlación. Opciones: iv,ks,roc,aucpr.
- training_period: Rango de entrenamiento para el CSI estático. En caso de no tener, especificar `None`.
- verbose: Booleano que indica si se imprimen mensajes de progreso en la consola.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, siga los siguientes pasos:

1. Haga un clon del repositorio.
2. Cree una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realice sus cambios y haga commit (`git commit -am 'Añadir nueva funcionalidad'`).
4. Haga push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Cree un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulte el archivo `LICENSE` para más detalles.
