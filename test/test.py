import sys
import pandas as pd
import numpy as np
from CREDENTIALS import credentials
sys.path.append(credentials['path'])
from data.data import create_simulated_data

df = create_simulated_data(n_features=10,n_rows=1_000)
df['event_time'] = df['event_time'].astype(str)
features = [col for col in df.columns if col not in ['target','event_time','cliente_id']]
nan_fraction = 0.1
nan_indices = np.random.choice(df.index, size=int(len(df) * nan_fraction), replace=False)
for col in features:
  df.loc[nan_indices, col] = np.nan

parametros_analisis = {
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
}

from MLToolKit.feature_analysis import FeatureAnalysis
fa = FeatureAnalysis(df,parametros_analisis)
fa.univariado()
fa.estabilidad()
fa.correlacion_por_metrica()
fa.consolidar_analisis()
fa.save_files()