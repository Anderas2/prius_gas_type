
At the gas station, i have the habit switch between SP98 and E10. E10 is sold around 10 cents less expensive, however, the car consumes more of it per 100km. E10 contains 10% alcohol and is otherwise "super" fuel, sold as "95" in some countries. SP98 is the fuel sold as "super plus" or "super 98".

My question is: Is this higher consumption of E10 eating the better price or not? Asked the other way round: Is E10 fuel in the end really less expensive or not?

This consumption difference between two fuels is difficult to find because my car uses more or less gas depending on the weather, the traffic conditions, my personal mood, the speed, and the length of the route. For this first try, i did not connect to the CAN bus, so i had no information about the motor temperature and only one measurement per ride, taken by hand. As if it was not difficult enough, the Prius needs only one refill per month, so the season was changing while i did the recording.

I orient myself on an R script of [Victor Chernozhukov](http://www.mit.edu/~vchern/); who was beautifully extracting the influence on being female on the salary. However, he used R which i don't know, so i try to repeat this in python. 


```python
import statsmodels.api as sm
import pandas as pd
import numpy as np
from patsy import dmatrices
```

    C:\Users\Andreas\Anaconda3\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools
    


```python
df = pd.read_excel('measurements.xlsx')
df.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>distance</th>
      <th>consume</th>
      <th>speed</th>
      <th>temp_inside</th>
      <th>temp_outside</th>
      <th>specials</th>
      <th>gas_type</th>
      <th>AC</th>
      <th>rain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28.0</td>
      <td>5.0</td>
      <td>26.0</td>
      <td>21.5</td>
      <td>12</td>
      <td>NaN</td>
      <td>E10</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.0</td>
      <td>4.2</td>
      <td>30.0</td>
      <td>21.5</td>
      <td>13</td>
      <td>NaN</td>
      <td>E10</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.2</td>
      <td>5.5</td>
      <td>38.0</td>
      <td>21.5</td>
      <td>15</td>
      <td>NaN</td>
      <td>E10</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.9</td>
      <td>3.9</td>
      <td>36.0</td>
      <td>21.5</td>
      <td>14</td>
      <td>NaN</td>
      <td>E10</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.5</td>
      <td>4.5</td>
      <td>46.0</td>
      <td>21.5</td>
      <td>15</td>
      <td>NaN</td>
      <td>E10</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



* distance is the distance in kilometers i was driving
* consume is the consumption in liters per 100 kilometers as seen in the display
* speed is the average speed. 
* temp_inside is the setting of the heating or "NaN" if it was turned off
* temp_outside is the temperature outside, taken at the end of the ride.
* specials is a remark if it was raining, snowing or if the climatization was on ("AC")
* gas type is the gas type used during the last refill
* AC is one hot encoded, the special "AC". 1 for on, 0 for off.
* rain is one-hot-encoded, the special "rain" and "snow". 1 for it was raining/snowing, 0 for it was good weather.


```python
# if the heating was turned completely off, replace the inside temperature by the outside temperature
df['temp_inside'].fillna(df['temp_outside'], inplace=True)
# get the temperature difference
df['temp_diff'] = df['temp_inside'] - df['temp_outside']
# add the square of the speed to the frame
df['speedsquare'] = df['speed']**2

# translate the gas type to something machine readable
def gastype(in_string):
    '''gas type in, integer out'''
    if in_string == "E10":
        return 0
    else:
        return 1
df['gas_type_num']= df['gas_type'].apply(gastype)
df.groupby(by='gas_type')['consume'].mean()
```




    gas_type
    E10     5.193750
    SP98    5.277174
    Name: consume, dtype: float64



So yes there is indeed an impact of the gas type. Contrary to my real life experience it looks as if SP98 makes my car consume more! This is because i used SP98 throughout the winter while E10 was used before and after, unintentionally.

As i am fairly confident with sklearn by now, first i tried to do the inference with sklearn regression analyses.

Basically, if you want to extract the influence of one feature, you remove it from the feature space together with the target variable. 
Then you fit the regression on the target variable, `consume` in my case, and a second regression on the feature to be extracted: `gas_type` in my case.

Then comes a step that i don't fully understand: You fit the coefficients of the two regressions on each other, and the outcome shall be a number in the unit of the target variable, depicting the result.
Could you check the marked part below? I don't trust myself here!


```python
# make numpy vectors for prediction
prediction_values = ['distance', 'speed', 'speedsquare', 'temp_diff', 'AC', 'rain']
all_values = ['gas_type_num','consume','distance', 'speed', 'speedsquare', 'temp_diff', 'AC', 'rain']

X = df[prediction_values].values
Y = df['consume'].values
Y_gas = df['gas_type_num'].values

# apply regression
from sklearn.linear_model import LinearRegression
rgr = LinearRegression()
rgr.fit(X, Y)

# apply again, this time trained on gas type
rgr_gas = LinearRegression()
rgr_gas.fit(X, Y_gas)

#####################################################################################
# do inference: This is the part that i don't understand! Could you please help here?
rgr_inference = LinearRegression()
rgr_inference.fit(rgr.coef_.reshape(-1,1), rgr_gas.coef_)
difference = rgr_inference.coef_[0]



print('\nThe result after crossfitting two regressions to get the effect of gas sorts:')
print('The difference in consumption between E10 and SP98 is {:.2f} liter.'.format(difference))
print('Assuming a price difference of 10 Cents, E10 = 1,40€ and SP98 = 1,50€')
low_consume = df['consume'].mean() *1.5
high_consume = (difference + df['consume'].mean()) * 1.4
price_difference = high_consume - low_consume
print('it means that 100km cost {:.2f} cents more with the supposedly cheaper E10.'.format(price_difference))

print('\n\nThe importance of the other factors (F-Values)')
from sklearn.feature_selection import f_regression
F, pval = f_regression(X, Y)
predictors_df = pd.DataFrame(columns=prediction_values)
predictors_df.loc['predictors'] = F
print(predictors_df.round(2))
```

    
    The result after crossfitting two regressions to get the effect of gas sorts:
    The difference in consumption between E10 and SP98 is 0.84 liter.
    Assuming a price difference of 10 Cents, E10 = 1,40€ and SP98 = 1,50€
    it means that 100km cost 0.65 cents more with the supposedly cheaper E10.
    
    
    The importance of the other factors (F-Values)
                distance  speed  speedsquare  temp_diff    AC  rain
    predictors      0.68   5.32         2.34       4.95  2.01  2.71
    

So far how it *should* work. Sadly, i am absolutely not sure about the second step, where the two coefficient vectors were fitted to each other. Also, sklearn does not offer confidence intervals - so i don't know how reliable this result really is - or if it is a result at all.


```python

```
