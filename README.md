
<img src="gas_station_orig.jpg">


At the gas station, i have the habit switch between SP98 and E10. E10 is sold around less expensive, however, the car consumes more of it per 100km. By feeling i would say it is between 0.5 and 1 liter more per 100km - which is, taken by logic, ridiculous lots.

I want to try and find the real impact on the consumption today. My question is: Is this higher consumption of E10 eating the better price or not? Asked the other way round: Is E10 fuel in the end really less expensive or not?


```python
E10_price = 1.379
SP98_price = 1.459
```

E10 contains 10% alcohol and is otherwise "super" fuel, sold as "95" in some countries. SP98 is the fuel sold as "super plus" or "super 98".

This consumption difference between two fuels is difficult to find because my car uses more or less gas depending on the weather, the traffic conditions, my personal mood, the speed, and the length of the route. For this first try, i did not connect to the CAN bus, so i had no information about the motor temperature and only one measurement per ride, taken by hand. As if it was not difficult enough, the Prius needs only one refill per month, so the season was changing while i did the recording.

I orient myself on an R script of [Victor Chernozhukov](http://www.mit.edu/~vchern/); who was beautifully extracting the influence on being female on the salary. However, he used R which i don't know, so i try to repeat this in python. 


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from patsy import dmatrices
```


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
# indicator if the heating was not used at all
df['heating_off']=df['temp_inside'].isnull()
# if the heating was turned completely off, replace the inside temperature by the outside temperature
df['temp_inside'].fillna(df['temp_outside'], inplace=True)
# get the temperature difference
df['temp_diff'] = df['temp_inside'] - df['temp_outside']
df['temp_diff_square'] = df['temp_diff']**2
# add the square and cube of the speed to the frame
df['speedsquare'] = df['speed']**2  # 5% better accuracy
df['speedcube'] =  df['speed']**3  # 1% better accuracy

# add an indicator for the heat up phase to the frame
# it is a timer that measures roughly 15 minutes from start indirectly via distance and speed,
# and then uses a sigmoid function to cut off. 5% better accuracy
df['startphase'] = 1 / (1 + np.exp( ((df['distance']/(df['speed']/12)) -3.3)/0.4 ))

# heating costs extra in the startphase, later not so much
df['start_heating'] = df['startphase'] * df['temp_diff'] #0.3% better accuracy

# translate the gas type to something machine readable
def gastype(in_string):
    '''gas type in, integer out'''
    if in_string == "E10":
        return 0
    else:
        return 1
df['gas_type_num']= df['gas_type'].apply(gastype)
print(df.groupby(by='gas_type')['consume'].mean().round(2))
```

    gas_type
    E10     5.18
    SP98    5.28
    Name: consume, dtype: float64
    

So yes there is indeed an impact of the gas type. Contrary to my real life experience it looks as if SP98 makes my car consume more, and that the difference is very small! This is because i used SP98 throughout the winter, while E10 was used before and after. I need to refill only once per month so it is difficult to have both gas types in the same season.

As i am fairly confident with sklearn by now, first i tried to do the inference with sklearn regression analyses.

Basically, if you want to extract the influence of one feature, you remove it from the feature space together with the target variable. 
Then you fit the regression on the target variable, `consume` in my case, and a second regression on the feature to be extracted: `gas_type` in my case.

Then comes a step that i don't fully understand: You fit the coefficients of the two regressions on each other, and the outcome shall be a number in the unit of the target variable, depicting the result.
Could you check the marked part below? I don't trust myself here!


```python
# make numpy vectors for prediction
prediction_values = ['distance','start_heating', 'startphase', 
                     'speed', 'speedsquare', 'speedcube', 
                     'temp_diff', 'temp_diff_square', 'temp_outside', 
                     'heating_off', 'AC', 'rain']

X = df[prediction_values].values
Y = df['consume'].values
Y_gas = df['gas_type_num'].values

# apply regression
rgr = LinearRegression()
rgr.fit(X, Y)

# apply again, this time trained on gas type
rgr_gas = LinearRegression()
rgr_gas.fit(X, Y_gas)

# get the residuals (the not-yet-explained variance left in the data)
Y_residuals = Y - rgr.predict(X)
X_gas_residuals = Y - rgr_gas.predict(X)

# fit the residuals to get the influence of the gas type
# reshape(-1,1) is necessary since scikit 19 if you have a single feature
rgr_inference = LinearRegression()
rgr_inference.fit(X_gas_residuals.reshape(-1,1), Y_residuals)
difference = rgr_inference.coef_[0] # there is only one coef, but given as list of one. :-)

print('\nThe result after crossfitting two regressions to get the effect of gas sorts:')
print('The difference in consumption between E10 and SP98 is {:.2f} liter.'.format(difference))


# out of interest - what was the influence of the other factors?
print('\n\nThe importance of the other factors (F-Values)')
from sklearn.feature_selection import f_regression
F, pval = f_regression(X, Y)
predictors_df = pd.DataFrame(columns=prediction_values)
predictors_df.loc['F-value of predictor'] = F
print(predictors_df.round(2).transpose())
print('\nAnd R² of the model:{:.3f}'.format(rgr.score(X, Y)))
```

    
    The result after crossfitting two regressions to get the effect of gas sorts:
    The difference in consumption between E10 and SP98 is 0.70 liter.
    
    
    The importance of the other factors (F-Values)
                      F-value of predictor
    distance                          1.61
    start_heating                    17.06
    startphase                       11.01
    speed                             9.39
    speedsquare                       4.16
    speedcube                         1.64
    temp_diff                         3.49
    temp_diff_square                  2.70
    temp_outside                      2.78
    heating_off                       0.73
    AC                                1.27
    rain                              4.82
    
    And R² of the model:0.368
    

So far how it *should* work. 

One problem remains: `sklearn` is tuned to predictions and does not offer confidence intervals - so i don't know how reliable this result really is - or if it is a result at all. So i head over to `statsmodels` now.


```python
# prepare dataframe for statsmodels
residuals = pd.DataFrame(Y_residuals, columns=['consume'])
residuals['E10']=X_gas_residuals

# fit regression in statsmodels format.
# it's like sklearn rgr.fit(E10, consume)
results = smf.ols('consume ~ E10', data=residuals).fit()

# get the result out of the vast array of available values
consume = results.conf_int().loc['E10']

print("The car uses between {:.2f} and {:.2f} L/100km more gas with E10, with 95% confidence".format(
            consume[0], consume[1]))

# assuming the difference is the beta of E10: attention, this
# is true only if the factor E10 is completely independent
difference = results.params[1]

#output results
print("\nThe mid point is {:.2f}L/100km".format(difference))
print('Here, E10 costs {:1.2f}€ and SP98 costs {:1.2f}€.\n\nSo, for 100 km:'.format(E10_price, SP98_price))
print('E10  consume: {:.2f}L and cost {:.3f}€'.format(df['consume'].mean() + (difference/2), 
                                                   df['consume'].mean() + (difference/2) * E10_price))
print('SP98 consume: {:.2f}L and cost {:.3f}€'.format(df['consume'].mean() - (difference/2), 
                                                   df['consume'].mean() - (difference/2) * SP98_price))
```

    The car uses between 0.62 and 0.77 L/100km more gas with E10, with 95% confidence
    
    The mid point is 0.70L/100km
    Here, E10 costs 1.38€ and SP98 costs 1.46€.
    
    So, for 100 km:
    E10  consume: 5.58L and cost 5.717€
    SP98 consume: 4.89L and cost 4.727€
    

That's a pretty clear result. 

Influence factor partialling out is a really cool feature of linear regression! :-)

I hope you had some fun reading this notebook - 

I will continue now and see how E80 gas behaves in my car. The measurement will again take some months, so don't wait in front of your screen for it. :-)

<img src="gas_station_correct.jpg">


```python

```
