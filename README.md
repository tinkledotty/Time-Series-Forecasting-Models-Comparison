# <b style="color:Gray;">1 <span style='color:#F1C40F'>|</span>  Introduction</b>

<div style="color:white;display:fill;border-radius:8px;
            background-color:#b4a7d6;font-size:150%;
            font-family:Nexa;letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>1.1 | About</b></p>
</div>

A noob first attempt in machine learning on TS data

**Objective:** Compare traditional forecasting methods with machine learning models on one of the top selling product family <br><br><br><br>



# <b style="color:Gray;">2 <span style='color:#F1C40F'>|</span> Prepare</b>

<div style="color:white;display:fill;border-radius:8px;
            background-color:#b4a7d6;font-size:150%;
            font-family:Nexa;letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>2.1 | Import Libraries</b></p>
</div>
To import CSV file into DataFrame:

```
import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta
import os
from itertools import cycle

# visualisation
import matplotlib.pyplot as plt
%matplotlib inline
#plt.rcParams["figure.figsize"] = (18,6)
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import matplotlib

# statistics
import statsmodels.api as sm
#import statsmodels.tsa.api as smt
#import statsmodels.formula.api as smf



from pmdarima import auto_arima
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
#from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb

from statsmodels.graphics.tsaplots import plot_acf


from warnings import filterwarnings, simplefilter
filterwarnings('ignore')
simplefilter('ignore')
```

```python
# Reading csv file into DataFrame
train = pd.read_csv('train.csv', parse_dates=True, infer_datetime_format = True)
test = pd.read_csv('test.csv', parse_dates=True, infer_datetime_format = True)
stores = pd.read_csv('stores.csv')
oil = pd.read_csv('oil.csv', parse_dates=True, infer_datetime_format = True)
transactions =  pd.read_csv('transactions.csv', parse_dates=True, infer_datetime_format = True)
holidays = pd.read_csv('holidays_events.csv', parse_dates=True, infer_datetime_format = True)
sample_submission = pd.read_csv('sample_submission.csv')
```

```python
df_data = train.merge(stores, how="left", on='store_nbr')   
df_data = df_data.merge(oil, how="left", on='date')      
df_data = df_data.merge(transactions, how="left", on=['date','store_nbr'])  
df_data = df_data.merge(holidays,on='date',how='left')
df_data = df_data.rename(columns={'type_x' : 'store_type','type_y':'holiday_type'})

```
```python
<div style="color:white;display:fill;border-radius:8px;
            background-color:#b4a7d6;font-size:150%;
            font-family:Nexa;letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>2.3 | Inspect Data</b></p>
</div> 

dates = pd.date_range(start=min(train['date']),end=max(train['date']))
fig = go.Figure(data=[go.Table(
    columnorder = [1,2],
    columnwidth = [250,500],
    header=dict(values=['<b>Description</b>', '<b>Value</b>'],
                line_color='darkslategray',
                fill_color='mediumpurple',
                align=['left','center'],
                font=dict(color='white', size=12)),

    cells=dict(values=[['Data Start Date:', 
                        'Data End Date:',
                        'Number of rows:', 
                        'Total Days of sales:',
                        'Total number of days:',
                        'Number of Stores:', 
                        'Number of Product Families:', 
                        'Number of Cities:', 
                        'Number of States:',
                        'Number of cluster:',
                        'Store type:',
                        'Holiday type:',
                        'Locale:'],
                       [df_data['date'].min(), 
                        df_data['date'].max(),
                        df_data.shape[0], 
                        df_data['date'].nunique(),
                        len(dates),
                        df_data['store_nbr'].nunique(), 
                        df_data['family'].nunique(),
                        df_data['city'].nunique(),
                        df_data['state'].nunique(),
                        df_data['cluster'].nunique(),
                        df_data['store_type'].unique(),
                        df_data['holiday_type'].unique(),
                        df_data['locale'].unique()]],
               line_color='darkslategray',
               fill=dict(color=['ivory', 'white']),
               align=('left')))
               fill_color='lightgrey'))
])

fig.update_layout({"title": f'BASIC INFO OF DATA'}, height=600, width=800)
fig.show()
```
![newplot (13)](https://user-images.githubusercontent.com/54738409/203565313-9e4a45ca-ffda-4c9b-81a0-f43bd162a7d0.png)


References:<br>

* https://www.analyticsvidhya.com/blog/2022/06/time-series-forecasting-using-python/
* https://www.kaggle.com/code/javigallego/time-series-forecasting-tutorial#4-%7C-Time-Series-Components
* https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide
* https://towardsdatascience.com/forecasting-wars-classical-forecasting-methods-vs-machine-learning-4fd5d2ceb716
