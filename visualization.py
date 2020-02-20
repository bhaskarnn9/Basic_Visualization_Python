#% import statements
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

#% data-exploration (optional)
# pd.options.display.html.table_schema = True
# pd.options.display.max_rows = None

#% read csv
df = pd.read_csv('input/diabetes.csv')

#% list of features
col_list = df.columns
col_list

#% datatypes
datatypes = df.dtypes
datatypes

#% dimensions of data
data_shape = df.shape
print(f'shape of data: {data_shape}')

#% simple EDA
df.describe()

# simle EDA2
df.describe(include='all').transpose()

#% Univariate Analyisis -Numerical- Histogram and boxplot and distplot

#% histogram
preg_hist = df['Pregnancies'].hist()
preg_hist.set_ylabel('Count')
preg_hist.set_xlabel('Pregnancies')

#% boxplot
bp_boxplot = df.boxplot(column='BloodPressure')
bp_boxplot

bp_boxplot = df.boxplot(column='BloodPressure', grid=False)
bp_boxplot

# distplot
sns.set_style("whitegrid")
sns.distplot(df.Pregnancies, kde=True)
plt.title('Histogram of Pregnancies')
plt.show();

#% Univariate Analyisis -Categorical- Bar chart and Pie chart
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None
df_sf_crime = pd.read_csv('input_sf_crime/train.csv')
df_sf_crime.head()

df_sf_crime.dtypes

df_sf_crime['Descript'].astype('cat')
sns.set_style('darkgrid')
sns.countplot('')
