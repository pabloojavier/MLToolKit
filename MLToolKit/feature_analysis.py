import pandas as pd
import os
from .univariate import univariateIV, xgboost_analysis
from .utils.utils import merge_sets
from .stability import stability_stat, csi_stat, get_features_with_low_variability
from typing import List,Any,Dict,Tuple

class FeatureAnalysis:
    def __init__(
            self,
            df : pd.DataFrame,
            params : Dict[str, Any]
    ):
        """
        Initializes the feature analysis with the given dataframe and parameters.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe containing the data to be analyzed.
        
        params : dict 
            A dictionary of parameters which must include:
            * xgboost_params: A dictionary of parameters to be passed to the XGBoost model to do the univariate analysis.
            * experiment_name: The name of the experiment to be created in MLFlow. Is recommended to use the local work directory.
            * nombre_reporte: The name of the report to be saved.
            * databricks: A boolean indicating if the code is running on Databricks.
            * features: List of feature names to be analyzed.
            * target_name: The name of the target variable.
            * cliente_id: The name of the client ID column.
            * periodo_id: The name of the period ID column.
            * threshold_low_variabilty: A float indicating the threshold for low variability. If somme feature has a variability lower than this threshold, it will be informed.
            * fill_na: The value to be used to fill missing values. This fill value will be used in the univariate analysis.
            * threshold_correlation: A float indicating the threshold for family correlation. If the correlation between two features is higher than this threshold, the feature with lower metric will be removed.
            * correlation_metric: The metric to be used to compare the features and drop one. Options are 'iv', 'ks', 'roc' and 'aucpr'.


        Raises
        ------
        AssertionError
            If 'features' is not in params or if it contains no variables.
            If any of the features are not numeric.

        Examples
        -------
        >>> df = pd.DataFrame(your_data)
        >>> params = {
        ...     'xgboost_params' : {"random_state": 42, "max_depth": 2},
        ...     'experiment_name' : <your_local_work_directory_on_databricks>,
        ...     'nombre_reporte' : 'local_test',
        ...     'databricks' : True,
        ...     'features': ['feature1', 'feature2'],
        ...     'target_name': 'target',
        ...     'cliente_id': 'cliente_id',
        ...     'periodo_id': 'periodo_id',
        ...     'threshold_low_variabilty': 0.95,
        ...     'fill_na': -9e8,
        ...     'threshold_correlation': 0.5,
        ...     'correlation_metric': 'ks'
        ... }
        >>> fa = FeatureAnalysis(df, params)
        >>> fa.univariado()
        >>> fa.save_files()
        """ 

        no_numerical_features = [col for col in params['features'] if not pd.api.types.is_numeric_dtype(df[col])]
        assert 'features' in params and len(params['features']) > 0, "Parámetro 'features' debe contener al menos una variable"
        assert len(no_numerical_features) == 0 , f'Hay variables no numéricas: {no_numerical_features}'
        assert not (params['databricks'] and not params.get('experiment_name', "")), "Si 'databricks' es True, 'experiment_name' no puede ser vacío. Recomiendo que sea el directorio de trabajo local en Databricks"
        assert 'databricks' in params.keys(), "El parámetro 'databricks' debe estar presente en los paramámetros."
        assert 'target_name' in params.keys(), "El parámetro 'target_name' debe estar presente en los parámetros."
        assert 'cliente_id' in params.keys(), "El parámetro 'cliente_id' debe estar presente en los parámetros."
        assert 'periodo_id' in params.keys(), "El parámetro 'periodo_id' debe estar presente en los parámetros."
        assert params['correlation_metric'].lower() in ['ks', 'roc', 'iv', 'aucpr'], "Metric must be one of ['ks', 'roc', 'iv', 'aucpr']"
        assert pd.api.types.is_datetime64_any_dtype(df[params['periodo_id']]), f"La columna {params['periodo_id']} debe ser de tipo datetime. Ejecutar df['{params['periodo_id']}'] = pd.to_datetime(df['{params['periodo_id']}'])"
        assert all(col in df.columns for col in [params['target_name'], params['cliente_id'], params['periodo_id']]), f"Las columnas {params['target_name']}, {params['cliente_id']} y {params['periodo_id']} deben estar presentes en el dataframe."

        self.params = params
        default_params = {
            'xgboost_params': {"random_state": 42, "max_depth": 2},
            'nombre_reporte': 'default_report',
            'threshold_low_variabilty': 0.95,
            'fill_na': -9e8,
            'threshold_correlation': 0.5,
            'correlation_method': 'pearson',
            'correlation_metric': 'ks',
            'training_period' : None,
            'univ_params' : {"random_state": 42, "max_depth": 2, "min_samples_leaf": 0.05},
        }
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        exclude_cols = [params['target_name'],params['cliente_id'],params['periodo_id']]
        self.df : pd.DataFrame = df[exclude_cols+params['features']]
        self.params['exclude_cols'] = exclude_cols

        self.dict_runs = {
            'univariado': False,
            'estabilidad': False
        }

    def univariado(self)->pd.DataFrame:
        """
        Perform the univariate analysis, with metrics such as IV, KS, ROC and AUCPR.

        Outputs
        -------
            pd.DataFrame
        """
        df_univ = univariateIV(
            self.df,
            model_params=self.params['univ_params'],
            report_name=f'reporte_univariado_{self.params["nombre_reporte"]}.html',
            exclude_cols=self.params['exclude_cols'],
            target_model=self.params['target_name'],
            fill_na=self.params['fill_na']
        )
        
        xgboost_params = {
            'objective' : 'binary:logistic',
            'eval_metric' : 'logloss',
            **self.params['xgboost_params']
        }
        
        df_xgboost = xgboost_analysis(
            df=self.df,
            target_value=self.params['target_name'],
            features=self.params['features'],
            fillna=self.params['fill_na'],
            params=xgboost_params
        )

        self.df_univ = pd.merge(
            df_xgboost,
            df_univ.rename(columns={'Information Value':'iv'}),
            on='Feature',
            how='outer'
        ).rename(columns={'Feature':'feature'})
        
        self.dict_runs['univariado'] = True
        return self.df_univ

    def estabilidad(self) -> Dict[str, pd.DataFrame]:
        """
        Perform the stability analysis and return the results in a dictionary.

        Outputs
        -------
        dict
            A dictionary containing the following
            * dynamic_csi: A dataframe containing the dynamic CSI for each feature.
            * weight_csi: A dataframe containing the weight of each feature in the CSI.
            * low_variability: A dataframe containing the features with low variability.
            * mean: A dataframe containing the mean of each feature.
            * std: A dataframe containing the standard deviation of each feature.
            * nulls: A dataframe containing the percentage of nulls of each feature.
        """
        self.df_stability = stability_stat(self.df,self.params['periodo_id'],self.params['features'])
        self.df_dynamic_csi = csi_stat(self.df,self.params['periodo_id'],self.params['features'])
        self.df_static_csi  = csi_stat(self.df,self.params['periodo_id'],self.params['features'],self.params['training_period'])

        population = self.df.groupby([self.params['periodo_id']]).agg({self.params['cliente_id']: "count"}).T
        new_df_csi = self.df_dynamic_csi.drop(columns=["quantile", "status", "status_2"])
        periods_csi = [i for i in list(new_df_csi.columns)]
        population = population[periods_csi]
        weights = population.loc[self.params['cliente_id'], :].values / sum(population.loc[self.params['cliente_id'], :].values)
        new_df_csi = (new_df_csi * weights).sum(axis=1) / sum(weights)
        self.new_df_csi = new_df_csi.to_frame().reset_index().rename(columns={"index":"feature",0:"weight_csi"})

        df_low_variability = get_features_with_low_variability(self.df,self.params['features'],self.params['threshold_low_variabilty'])
        self.df_low_variability = pd.DataFrame(df_low_variability,columns=['feature']).assign(low_variability = 'True')
        
        self.dict_runs['estabilidad'] = True
        dict_to_return = {
            'dynamic_csi': self.df_dynamic_csi,
            'static_csi': self.df_static_csi,
            'weight_csi': self.new_df_csi,
            'low_variability': self.df_low_variability,
            'mean':self.df_stability['mean'],
            'std':self.df_stability['std'],
            'nulls_cnt':self.df_stability['nulls_cnt'],
            'nulls_pct':self.df_stability['nulls_pct'],
            'over_three_std' : self.df_stability['over_three_std'],
            'under_three_std' : self.df_stability['under_three_std'],
            'over_five_std' : self.df_stability['over_five_std'],
            'under_five_std' : self.df_stability['under_five_std'],
        }
        return dict_to_return
    
    def correlacion_por_metrica(self)->Tuple[Dict[str,List[str]],List[str]]:
        """
        Perform the correlation analysis and return the selected and discarded features.

        Outputs
        -------
        Tuple
            A tuple containing:
            * A dictionary containing the selected features for each family.
            * A list of discarded features.
        """
        if self.dict_runs['univariado'] == False:
            print("No se ha ejecutado el análisis univariado. Ejecute FeatureAnalysis.univariado()")
            return None,None
        
        metric = self.params['correlation_metric']
        threshold_correlation = self.params['threshold_correlation']
        correlation_method = self.params['correlation_method']

        familias = set([col.split('_')[0] for col in self.params['features'] if '_' in col])

        variables_seleccionadas = {}
        variables_descartadas = []
        
        for familia in familias:
            columnas_familia = [col for col in self.df.columns if col.split('_')[0] == familia]
            df_familia = self.df[columnas_familia]
            correlacion = df_familia.corr(method=correlation_method).abs()
            sets_correlacionados = []
            for col in correlacion.columns:
                correlacionadas = set(correlacion.index[correlacion[col] > threshold_correlation])
                if len(correlacionadas) > 1:
                    sets_correlacionados.append(correlacionadas)
            
            sets_correlacionados = merge_sets(sets_correlacionados)
            for conjunto in sets_correlacionados:
                df_conjunto = self.df_univ.loc[self.df_univ['feature'].isin(conjunto)].copy()
                mejor_variable = df_conjunto.loc[df_conjunto[metric].idxmax(), 'feature']
                if familia not in variables_seleccionadas:
                    variables_seleccionadas[familia] = []
                variables_seleccionadas[familia].append(mejor_variable)
                variables_descartadas.extend([var for var in conjunto if var != mejor_variable])
        self.variables_seleccionadas = variables_seleccionadas
        self.variables_descartadas = variables_descartadas
        self.df_seleccionadas_corr = pd.DataFrame([var for sublist in variables_seleccionadas.values() for var in sublist],columns=['feature']).assign(is_selected_by_correlation = 'True')

        return variables_seleccionadas,variables_descartadas

    def consolidar_analisis(self):
        """
        Consolidate the results of the univariate and stability analysis in a single dataframe. Also, creates an Excel file with the results.
        """
        if any(not value for value in self.dict_runs.values()):
            print("Faltan análisis. Ejecute FeatureAnalysis.univariado() o FeatureAnalysis.estabilidad()")
            return None
        
        df_out = (
            self.df_dynamic_csi.reset_index().rename(columns={'index':'feature','status_2':'stability_index'})[['feature','stability_index']]
            .merge(self.new_df_csi,on='feature',how='left')
            .merge(self.df_univ,on='feature',how='left')
            .merge(self.df_low_variability,on='feature',how='left')
            .merge(self.df_seleccionadas_corr,on='feature',how='left')
            .assign(low_variability = lambda x: x['low_variability'].fillna('False'))
            .assign(is_selected_by_correlation = lambda x: x['is_selected_by_correlation'].fillna('False'))
        )

        path = f'./reporte_consolidado_{self.params["nombre_reporte"]}.xlsx'
        with pd.ExcelWriter(path) as writer:
            names = ['features','dynamic_csi','static_csi',*self.df_stability.keys()]
            for i, table in enumerate([df_out,self.df_dynamic_csi,self.df_static_csi,*self.df_stability.values()]):
                table.to_excel(writer, sheet_name=f'{names[i]}', index=True)
        return df_out
    
    def save_files(self):
        if self.params['databricks']:
            self.dbfs_save_files()
        else:
            self.local_save_files()

    def local_save_files(self):
        pass

    def dbfs_save_files(self):
        import mlflow
        import time
        mlflow.sklearn.autolog(disable=True)
        mlflow.xgboost.autolog(disable=True)
        mlflow.statsmodels.autolog(disable=True)
        time_id = time.strftime('%Y-%m-%d_%H-%M-%S').replace('-','')
        mlflow.set_experiment(self.params['experiment_name']+'_'+time_id)
        
        nombres = {
            'univariado': f'reporte_univariado_{self.params["nombre_reporte"]}.html',
            'estabilidad': f'reporte_consolidado_{self.params["nombre_reporte"]}.xlsx'
        }
        with mlflow.start_run():
            for key, value in self.dict_runs.items():
                if value:
                    mlflow.log_artifact(nombres[key])

        for ruta in nombres.values():
            if os.path.exists(ruta):
                os.remove("./"+ruta)

    def remove_files(self):
        nombres = {
            'univariado': f'reporte_univariado_{self.params["nombre_reporte"]}.html',
            'estabilidad': f'reporte_consolidado_{self.params["nombre_reporte"]}.xlsx'
        }
        for ruta in nombres.values():
            if os.path.exists(ruta):
                os.remove("./"+ruta)
        