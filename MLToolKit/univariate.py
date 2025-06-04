import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import roc_auc_score,roc_curve,average_precision_score

import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

import os
import shutil
import base64

from typing import List, Tuple

random_number = np.random.randint(10000, 99999)

def _extract_intervals(tree, feature):
    thresholds = []
    
    def dfs(node):
        nonlocal thresholds
        if tree.feature[node] == feature:
            thresholds.append((tree.threshold[node], tree.impurity[node]))
        if tree.children_left[node] != tree.children_right[node]:
            dfs(tree.children_left[node])
            dfs(tree.children_right[node])
    
    dfs(0)
    thresholds.sort(key=lambda x: x[0])
    if thresholds:
        impurity = [e[1] for e in thresholds]
        thresholds = [e[0] for e in thresholds]
        intervals = [float('-inf')] + thresholds + [float('inf')]
        if len(intervals)!= len(set(intervals)):
            return None,None
        return intervals, impurity
    return None, None

def _checkFolder(path=f'./plot_{random_number}'):
    path = Path(path)
    if not path.is_dir():
        os.mkdir(path)

def _calculateStats(df, feature, target_model):
    
    data = df.copy()
    
    summary_df = (
    data.groupby([feature, target_model])
        .agg({target_model: "count"})
        .rename(columns={target_model: "count"})
    )
    summary_df = summary_df.reset_index().pivot(
        index=feature, columns=target_model, values="count"
    )
    
    # Se cuentan los casos nulos existentes por tramo
    data["isna"] = data[feature].isna().astype(int)
    summary_df["Indet."] = (
        data.groupby([feature]).agg({"isna": "sum"}).values
    )
    
    # Se crea una columna con la totalidad de los casos por tramo
    summary_df["Total"] = summary_df.sum(axis=1).fillna(0)
    summary_df = summary_df.fillna(0).reset_index()
    
    # Cambiamos los nombres de las columnas
    summary_df.columns = ["Categoria", "Buenos", "Malos", "Indet.", "Total"]
    
    # Contamos la totalidad de buenos, malos y totalidad de datos existentes
    buenos_malos_suma = summary_df[["Buenos", "Malos", "Total"]].sum(axis=0)
    sum_total_malos = buenos_malos_suma["Malos"]
    sum_total_buenos = buenos_malos_suma["Buenos"]
    sum_total = buenos_malos_suma["Total"]
    
    # Se calculan porcentajes de interes
    summary_df["%BadRate"] = (summary_df["Malos"] * 100 / summary_df["Total"]).round(2)
    summary_df["%Buenos"] = (summary_df["Buenos"] * 100 / sum_total_buenos).round(2)
    summary_df["%Malos"] = (summary_df["Malos"] * 100 / sum_total_malos).round(2)
    summary_df["%Total"] = (summary_df["Total"] * 100 / sum_total).round(2)
    
    # Se calculan porcentajes acumulados y diferencia de los porcentajes
    # acumulados
    summary_df["%AB"] = summary_df["%Buenos"].cumsum().round(2)
    summary_df["%AM"] = summary_df["%Malos"].cumsum().round(2)
    summary_df["%AT"] = summary_df["%Total"].cumsum().round(2)
    summary_df["%Diff"] = np.abs(summary_df["%AM"] - summary_df["%AB"])
    
    # Se calcula el Weight of Evidence y Information Value por intervalo.
    summary_df["WoE"] = (
        np.log(
            (summary_df["Malos"] / sum_total_malos + 1e-5)
            / (summary_df["Buenos"] / sum_total_buenos + 1e-5)
        )
    ).round(4)
    summary_df["IV"] = (
        (
            (summary_df["Malos"] / sum_total_malos + 1e-5)
            - (summary_df["Buenos"] / sum_total_buenos + 1e-5)
        )
        * summary_df["WoE"]
    ).round(4)
    
    # Se calculan los totales de ciertas variables de interes
    add_row = pd.DataFrame(
        summary_df[["Buenos", "Malos", "Indet.", "Total", "IV"]].sum(axis=0)
    ).T
    add_row["%BadRate"] = (add_row["Malos"] * 100 / add_row["Total"]).round(2)
    add_row["Categoria"] = "Total"
    
    # Concatenamos la fila de totales a el dataframe de estadisticos
    return pd.concat([summary_df, add_row]).fillna(" ").reset_index(drop=True)

def _plotBarBadRateWoE(summaries, studyFeature):
    _checkFolder()
    summary = summaries[studyFeature].copy()
    x = summary.iloc[:-1, :]["Categoria"].astype(str)
    y1 = summary.iloc[:-1, :]["%BadRate"]
    y2 = summary.iloc[:-1, :]["WoE"]
    
    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.bar(x, y1, zorder=2, color='royalblue')
    plt.grid(True, axis='y', which='major', zorder=-1)
    plt.xlabel("% BadRate")
    plt.xticks(rotation = -45)

    plt.subplot(122)
    plt.bar(x, y2, zorder=2, color='royalblue')
    plt.grid(True, axis='y', which='major', zorder=-1)
    plt.xlabel("Weight of Evidence")
    plt.xticks(rotation = -45)
    plt.suptitle("Interesting Charts")

    plt.savefig(f"./plot_{random_number}/{studyFeature}_twoBars.png", bbox_inches='tight')
    plt.close()

def _getPlot(pattern, folder=f"./plot_{random_number}"):
    _checkFolder()
    return next(iter(Path(f"./plot_{random_number}").glob(pattern)))
    
def _loadPlot(plotPath):
    image = open(plotPath, "rb").read()
    return base64.b64encode(image).decode("utf-8")

def _createReport(df_table, dict_summaries, report_name='report'):
    """
    Create a HTML report that show the prediction univariate for
    each variable.
    """
    df_html = df_table.to_html()

    top5_famility = df_table.copy()
    top5_famility['Family'] = top5_famility['Feature'].str.split("_").str[0]
    top5_famility = (
        top5_famility.sort_values(by='Information Value',ascending=False)
        .groupby(['Family']).head(5)
        .sort_values(by=['Family'], ascending=False)
    )
    top5_famility = top5_famility.to_html()
    
    features_names = df_table["Feature"].values
    for feature_name in features_names:
        df_html = df_html.replace(
            f"<td>{feature_name}</td>\n",
            f'<td><a href="#{feature_name}">{feature_name}</a></td>\n',
        )
    df_html = df_html.replace(
        '<table border="1" class="dataframe">', '<table class="table table-striped">'
    )

    image_path = _getPlot(pattern="count_features.png")
    data_uri = _loadPlot(image_path)
    image_path = _getPlot(pattern="iv_features.png")
    data_uri_iv = _loadPlot(image_path)
    
    html_string = f"""
        <html lang="en">
        <head>
            <title>Reporte de Information Value</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
        </head>
        <body>
        <div class="container">
            <h1>Sección 1: Resultados Preliminares</h1>
            <a id="summary"></a>
            <p>En la siguientes pestañas podras encontrar información relacionada al dataset utilizado e nivel de predictivilidad de las variables según el Information Value (IV).</p>
        
            <ul class="nav nav-tabs">
            <li class="active"><a data-toggle="tab" href="#home">Tabla IV</a></li>
            <li><a data-toggle="tab" href="#menu1">Cant. de Variables</a></li>
            <li><a data-toggle="tab" href="#menu2">Predictibilidad por Familia</a></li>
            <li><a data-toggle="tab" href="#menu3">Top 5 por Familia</a></li>
            </ul>
        
            <div class="tab-content">
            <div id="home" class="tab-pane fade in active">
                <h3>Top 10 de variables con mayor nivel predictivo</h3>
                <p>Tabla de resumen ordenada de mayor a menor con las variables mas predictivas.</p>
                {df_html}
            </div>
            <div id="menu1" class="tab-pane fade">
                <h3>Cant. de Variables</h3>
                <p>Cantidad de variables analizadas agrupadas por familia.</p>
                <img src="data:image/png;base64,{data_uri}" class="center">
            </div>
            <div id="menu2" class="tab-pane fade">
                <h3>Predictibilidad por Familia</h3>
                <p>Predictibilidad de las variables según su information value agrupada por Familia. La línea roja señala el mínimo de predictibilidad exigido (0.02).</p>
                <img src="data:image/png;base64,{data_uri_iv}" class="center">
            </div>
            <div id="menu3" class="tab-pane fade">
                <h3>Top 5 por Familia</h3>
                <p>Top 5 predictibilidad agrupada por familia. La línea roja señala el mínimo de predictibilidad exigido (0.02).</p>
                {top5_famility.replace('<table border="1" class="dataframe">', '<table class="table table-striped">')}
                
            </div>
        <hr class="solid">
        <h1>Sección 2: Estadísticos por Variables</h1>
    """

    for it, feature_name in enumerate(features_names):
        df_summary = dict_summaries[feature_name].copy()
        df_summary = df_summary.to_html().replace(
            '<table border="1" class="dataframe">',
            '<table class="table table-striped">',
        )

        image_path_1 = _getPlot(f"{feature_name}_twoBars.png")
        data_uri_plot = _loadPlot(image_path_1)
        image_path_2 = _getPlot(f"{feature_name}_treeplot.png")
        data_uri_tree = _loadPlot(image_path_2)
        
        html_string += f"""
        <h3><b>ID: {it}</b></h3>
        <a id="{feature_name}"></a>
        <h3><b>Variable: {feature_name}</b></h3>
        <br>
        <center>
        {df_summary}
        </center>
        <ul class="nav nav-tabs">
        <li class="active"><a data-toggle="tab" href="#menuplot{feature_name}">BarPlots</a></li>
        <li><a data-toggle="tab" href="#menutree{feature_name}">Árbol</a></li>
        </ul>

        <div class="tab-content">
        <div id="menuplot{feature_name}" class="tab-pane fade in active">
            <h3>Gráficos de BadRate e Information Value</h3>
            <p>Comportamiento de las categorías con diferentes intervalos.</p>
            <img src="data:image/png;base64,{data_uri_plot}" class="center">
        </div>
        <div id="menutree{feature_name}" class="tab-pane fade">
            <h3>Árbol obtenido</h3>
            <p>Árbol generado durante el entrenamiento.</p>
            <img src="data:image/png;base64,{data_uri_tree}" class="center">
        </div>
        <hr class="solid">
        <br>
        <a href="#summary" class="btn btn-info" role="button">VOLVER AL INICIO</a>
        <hr class="solid">
        """

    html_string += """
    </div></body></html>"""
    f = open(f"./{report_name}", "w")
    f.write(html_string)
    f.close()

def _counter_vars(variables):
    counter = {}
    for f in variables:
        counter[f.split("_")[0]] = counter.get(f.split("_")[0], 0) + 1
    return pd.DataFrame(counter, index=[0]).T.rename(columns={0:'count'})

def _trainModel(X, y, 
               feature,
               params,
               model=tree.DecisionTreeClassifier
              ):
  
    clf = model(**params)
    clf.fit(X, y)
    fig = plt.figure(figsize=(10,5))
    _ = tree.plot_tree(clf)
    plt.savefig(f"./plot_{random_number}/{feature}_treeplot.png", bbox_inches='tight')
    plt.close()
    return clf

def _clearGarbage(path=f'./plot_{random_number}'):
    shutil.rmtree(Path(path))

def _ks_statistic(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    return ks

def xgboost_analysis(df, target_value, features,fillna=-9e8,params={},**kwargs):
  results = []
  it = 0
  for feature in features:
      if kwargs.get('verbose', True):
            print(f"Calculating KS/ROC: {(it+1):>4}/{len(features):<4} {feature:<80}",end="\r",flush=True)
      X = df[[feature]].fillna(fillna)
      y = df[target_value].astype(int)
      
      model = xgb.XGBClassifier(**params)
      model.fit(X, y)
      y_pred = model.predict_proba(X)[:, 1]
      
      roc = roc_auc_score(y, y_pred)
      ks = _ks_statistic(y, y_pred)
      auc = average_precision_score(y, y_pred)
      
      results.append({
          'Feature': feature,
          'roc': roc,
          'ks': ks,
          'aucpr' : auc
      })
      it+=1
  if kwargs.get('verbose', True):
    print()
  results_df = pd.DataFrame(results)
  return results_df

def univariateIV(dataset,
                 model_params={"random_state": 42, "max_depth": 2, "min_samples_leaf": 0.05},
                 report_name='reporte',
                 exclude_cols = ['mach_id', 'event_time', 'target_value'], 
                 target_model = 'target_value',
                 fill_na = -9e8,
                 **kwargs
                ):
    valid_cols = [i for i in dataset.columns if i not in exclude_cols+[target_model]]
    dict_summaries = {}
    _checkFolder()
    y = dataset[target_model]
    
    plot_count = _counter_vars(dataset[valid_cols]).sort_values(by='count', ascending=False)
    plt.barh(plot_count.index, plot_count['count'])
    plt.xticks(rotation = -90)
    plt.grid(True, axis='x', which='major', zorder=-1)
    plt.xlabel("Count of Features")
    plt.ylabel("Features Names")
    plt.savefig(f"./plot_{random_number}/count_features.png", bbox_inches='tight')
    plt.close()
    
    features_errors = []
    
    top_df = pd.DataFrame()
    for it,feature in enumerate(valid_cols):
        if kwargs.get('verbose', True):
            print(f"Calculating IV: {(it+1):>4}/{len(valid_cols):<4} {feature:<80}",end="\r",flush=True)
        X = dataset.loc[:, [feature]].fillna(fill_na)
        
        clf = _trainModel(X, y, feature=feature, params=model_params)
        intervals, impurity = _extract_intervals(clf.tree_, 0)
        
        if intervals is None:
            features_errors.append(feature)
            continue
        
        temporal_df = pd.DataFrame()
        temporal_df[feature] = pd.cut(X[feature], intervals)
        temporal_df[target_model] = y

        
        dict_summaries[feature] = _calculateStats(temporal_df, feature, target_model)
        _plotBarBadRateWoE(dict_summaries, feature)
        IV = dict_summaries[feature].iloc[-1, -1]
        top_new_row = pd.DataFrame({'Feature':feature, 'Information Value':IV}, index=[0])
        top_df = pd.concat([top_df, top_new_row])    
    if kwargs.get('verbose', True):
        print()
        print(f"Features with not enough variability: {len(features_errors)}")

    top_df = top_df.sort_values(by='Information Value', ascending=False).reset_index(drop=True)
    boxplot = pd.DataFrame()
    boxplot["variables"] = top_df['Feature'].str.split("_").str[0]
    boxplot["IV"] = top_df['Information Value']
    
    boxplot.pivot(columns='variables').boxplot(vert=False)
    plt.grid(True, axis='y', which='major', zorder=-1)
    plt.xlabel("Information Value")
    plt.xticks(rotation = -45)
    plt.xticks(np.arange(1, step=0.05))
    plt.axvline(0.02,color='red')
    plt.savefig(f"./plot_{random_number}/iv_features.png", bbox_inches='tight')
    plt.close()
    
    _createReport(top_df, dict_summaries, report_name)
    _clearGarbage()
    new_rows = pd.DataFrame({
      'Feature': features_errors,
      'Information Value': [0]*len(features_errors),
    })

    top_df = pd.concat([top_df, new_rows], ignore_index=True)
    return top_df
