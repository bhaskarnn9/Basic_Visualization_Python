# %

# import statements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# %

init_notebook_mode(connected=True)

# %

# read csv
df = pd.read_csv('input/diabetes.csv')
df_sf_crime = pd.read_csv('input/train.csv')

# %

# list of features
col_list = df.columns
print(col_list)

# %

# data-types
data_types = df.dtypes
print(data_types)

# %

# dimensions of data
data_shape = df.shape
print(f'shape of data: {data_shape}')

# %

# simple EDA
df.describe()

# %

# simple EDA2
df.describe(include='all').transpose()

# %

# Uni-variate Analysis -Numerical- Histogram and box-plot and dist-plot

# %

# histogram
plt.figure(figsize=(15, 8))
pregnancy_hist = df['Pregnancies'].hist()
pregnancy_hist.set_ylabel('Count')
pregnancy_hist.set_xlabel('Pregnancies')

# %

# box-plot
plt.figure(figsize=(10, 10))
bp_box_plot = df.boxplot(column='BloodPressure')
print(bp_box_plot)

# %

# box-plot
plt.figure(figsize=(10, 10))
bp_box_plot = df.boxplot(column='BloodPressure', grid=False)
print(bp_box_plot)

# %

# dist-plot
sns.set_style("whitegrid")
plt.figure(figsize=(15, 8))
sns.distplot(df.Pregnancies, kde=True)
plt.title('Histogram of Pregnancies')
plt.show()

# %

# Uni-variate Analysis -Categorical- Bar chart and Pie chart
df_sf_crime.head()

# %

print(df_sf_crime.dtypes)

# %

plt.figure(figsize=(15, 8))
sns.set_style('darkgrid')
sns.countplot(y='Category', data=df_sf_crime)

# %

plt.figure(figsize=(15, 8))
df_sf_crime['Category'].value_counts().plot(kind='bar')

# %

plt.figure(figsize=(15, 8))
df_sf_crime['Category'].value_counts().plot(kind='barh')

# %

plt.figure(figsize=(15, 15))
df_sf_crime['Category'].value_counts().plot(kind='pie')

# %

np.random.seed(0)  # Set seed for reproducibility
n = 10
r1 = np.random.randn(n)
r2 = np.random.randn(n)
print(min(r1))
print(max(r1))

trace0 = go.Box(
    y=r1,
    name='Box1',
    marker=dict(
        color='#AA0505',
    )
)
trace1 = go.Box(
    y=r2,
    name='Box2',
    marker=dict(
        color='#B97D10',
    )
)
data = [trace0, trace1]
layout = go.Layout(title="Box-plot of 2 sets of random numbers")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
