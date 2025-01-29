from setuptools import setup, find_packages

from MLToolKit import __version__

setup(
    name='MLToolKit',
    version=__version__,
    url='https://github.com/pabloojavier/MLToolKit',
    author='Pablo Gutiérrez',
    author_email='pgutierrez2018@udec.cl',
    packages=find_packages(),
    install_requires=[
        'mlflow==1.19.0',
        'xgboost==2.0.0',
        'shap==0.42.0',
        'scikit-learn==0.24',
        'pandas==1.2.4',
        'numpy==1.23.0',
        'openpyxl==3.1.5',
        'matplotlib>=3.6.0',
    ],
)