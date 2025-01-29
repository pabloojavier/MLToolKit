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
parametros = {
  'xgboost_params' : {"random_state": 42, "max_depth": 2},
  'experiment_name' : ruta_completa_a_tu_carpeta_de_desarrollo,
  'databricks' : True,
  'nombre_reporte' : 'variables_test',
  'target_name' : 'target',
  'periodo_id' : 'event_time',
  'cliente_id' : 'cliente_id',
  'features' : features,
  'threshold_low_variabilty' : 0.95,
  'fill_na' : -9e8, 
  'threshold_correlation' : 0.5,
  'correlation_metric' : 'aucpr', # Opciones: iv,ks,roc,aucpr
}

from MLToolKit.feature_analysis import FeatureAnalysis
fa = FeatureAnalysis(df,parametros)
fa.univariado()
fa.estabilidad()
fa.correlacion_por_metrica()
fa.consolidar_analisis()
fa.save_files()
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, siga los siguientes pasos:

1. Haga un clon del repositorio.
2. Cree una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realice sus cambios y haga commit (`git commit -am 'Añadir nueva funcionalidad'`).
4. Haga push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Cree un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulte el archivo `LICENSE` para más detalles.
