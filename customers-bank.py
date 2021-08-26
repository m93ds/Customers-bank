# Leemos archivo .zip desde la web
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

# Extraemos el contenido del .zip
! unzip /content/bank.zip

# Importamos la librería pandas para utlizar la función que lee .csv
import pandas as pd

# Leemos el archivo .csv
data=pd.read_csv('/content/bank-full.csv', sep=';')

# Observamos los primeros 10 valores del conjunto de datos
data.head(10)

# Vemos la cantidad de registros y columnas
data.shape

# Instalamos ProfileReport
! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

# Cargamos ProfileReport
from pandas_profiling import ProfileReport

# Aplicamos ProfileReport a la base de datos (data = bank-full.csv)
ProfileReport(data)

# Observamos los nombres de las columnas
data.columns

# Convertimos la variable job de string a numérica
data['job'].replace({"unemployed": 0, "student": 1, "blue-collar": 2, "housemaid": 3, "services": 4, "technician": 5, "admin.": 6, "management": 7,
                     "self-employed": 8, "entrepreneur": 9, "retired": 10, "unknown": None},  inplace = True)
                     
# Convertimos la variable marital de string a numérica
data['marital'].replace({"single": 0, "married": 1, "divorced": 2, "unknown": None},  inplace = True)

# Convertimos la variable education de string a numérica
data['education'].replace({"primary": 0, "secondary": 1, "tertiary": 2, "unknown": None},  inplace = True)

# Convertimos la variable contact de string a numérica
data['contact'].replace({"cellular": 0, "telephone": 1, "unknown": None},  inplace = True)

# Convertimos la variable poutcome de string a numérica
data['poutcome'].replace({"failure": 0, "success": 1, "other": 2, "unknown": None},  inplace = True)

# Revisamos los primeros 5 valores de la base de datos
data.head(5)

# Seteamos para que no devuelva decimales y volvemos a revisar la base de datos (Primeros 5 valores y últimos 5 valores)
pd.set_option('precision',0) 
data

# Verificamos la existencia de valores nulos
data.isnull().sum()

# Importamos las librerías necesarias para imputar los valores faltantes
import numpy as np
import sklearn 
from sklearn.impute import SimpleImputer

# Calculamos la moda que posteriormente será el valor imputado a los valores faltantes de la variable poutcome
data['poutcome'].mode() 

# Hacemos la imputación de la moda para la variable poutcome
imp=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data['poutcome']=imp.fit_transform(data[['poutcome']])

# Instalamos missingpy para aplicar KNN
!pip install missingpy

# Importamos la librería necesaria para imputar usando KNN
import sklearn 
from missingpy import KNNImputer

# Usamos KNN con el criterio de distancia para la variable job
imp=KNNImputer(n_neighbors=2, weights='distance')
data['job']=imp.fit_transform(data[['job']])

# Usamos KNN con el criterio de distancia para la variable education
imp=KNNImputer(n_neighbors=2, weights='distance')
data['education']=imp.fit_transform(data[['education']])

# Usamos KNN con el criterio de distancia para la variable contact
imp=KNNImputer(n_neighbors=2, weights='distance')
data['contact']=imp.fit_transform(data[['contact']])

# Redondeamos los valores de la base de datos para eliminar los decimales
data = round(data,0)

# Revisamos si hay valores nulos en el conjunto de datos
data.isnull().sum() 

# Observamos los primeros 10 valores de la base de datos
data.head(10)

# Exportamos los datos creando un csv
from google.colab import files
data.to_csv('bank_clean.csv', sep=";") 

# Descargamos csv con la base de datos limpia
files.download('bank_clean.csv')

# Leemos el archivo .csv limpio
datos=pd.read_csv('/content/bank_clean.csv', sep=';')

# Importamos la librería requerida para graficar
import seaborn as sb
from seaborn import barplot
from seaborn import violinplot
from seaborn import relplot

# Graficamos dos variables (Balance y job)
graf1 = sb.barplot(x='job', y='balance', data= data, estimator=lambda job: len(job) / len(data) * 100)
graf1.set(xlabel="Job")
graf1.set(ylabel="Balance")
# 0=unemployed; 1=student; 2=blue-collar; 3=housemaid; 4=services; 5=technician; 6=admin; 7=management; 8=self-employed; 9=entrepreneur; 10=retired

# Graficamos tres variables (Education, balance y loan)
graf2 = sb.barplot( x='education', y='balance', hue='loan', data=data, estimator=lambda education: len(education) / len(data) * 100)
graf2.set(xlabel="Education")
graf2.set(ylabel="Balance")
# 0=Primary; 1=Secundary; 2=Tertiary

#Graficamos tres variables (Marital, age y housing)
graf3=sb.boxplot(x='marital', y='age', hue='housing', data=data,saturation=1.2,width=0.8, fliersize=5, palette="Set2")
graf3.set(xlabel="Marital")
graf3.set(ylabel="Age")
# 0=single; 1=married, 2=divorced

# Graficamos tres variables (Poutcome, duration y contact)
graf4 = sb.violinplot(data=data, x='poutcome', y='duration', hue='contact', width=0.8, palette="Set1")
#poutcome (0=failure; 1=success; 2=other) ; contact (0=cellular; 1=telephone)
