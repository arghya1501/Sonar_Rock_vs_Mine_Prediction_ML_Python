## SOLAR ROCK AND MINE PREDICTION ML PROJECT


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## DATA COLLECTION AND DATA PROCESSING


```python
#LOADING THE DATASET TO PANDAS DATAFRAME
sonar_data = pd.read_csv(r"E:\Project work 2\Machine Larning\Sonar Rock vs Mine Prediction\Copy of sonar data.csv", header = None)
```


```python
#TOP 5 ROWS 
sonar_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0200</td>
      <td>0.0371</td>
      <td>0.0428</td>
      <td>0.0207</td>
      <td>0.0954</td>
      <td>0.0986</td>
      <td>0.1539</td>
      <td>0.1601</td>
      <td>0.3109</td>
      <td>0.2111</td>
      <td>...</td>
      <td>0.0027</td>
      <td>0.0065</td>
      <td>0.0159</td>
      <td>0.0072</td>
      <td>0.0167</td>
      <td>0.0180</td>
      <td>0.0084</td>
      <td>0.0090</td>
      <td>0.0032</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0453</td>
      <td>0.0523</td>
      <td>0.0843</td>
      <td>0.0689</td>
      <td>0.1183</td>
      <td>0.2583</td>
      <td>0.2156</td>
      <td>0.3481</td>
      <td>0.3337</td>
      <td>0.2872</td>
      <td>...</td>
      <td>0.0084</td>
      <td>0.0089</td>
      <td>0.0048</td>
      <td>0.0094</td>
      <td>0.0191</td>
      <td>0.0140</td>
      <td>0.0049</td>
      <td>0.0052</td>
      <td>0.0044</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0262</td>
      <td>0.0582</td>
      <td>0.1099</td>
      <td>0.1083</td>
      <td>0.0974</td>
      <td>0.2280</td>
      <td>0.2431</td>
      <td>0.3771</td>
      <td>0.5598</td>
      <td>0.6194</td>
      <td>...</td>
      <td>0.0232</td>
      <td>0.0166</td>
      <td>0.0095</td>
      <td>0.0180</td>
      <td>0.0244</td>
      <td>0.0316</td>
      <td>0.0164</td>
      <td>0.0095</td>
      <td>0.0078</td>
      <td>R</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0100</td>
      <td>0.0171</td>
      <td>0.0623</td>
      <td>0.0205</td>
      <td>0.0205</td>
      <td>0.0368</td>
      <td>0.1098</td>
      <td>0.1276</td>
      <td>0.0598</td>
      <td>0.1264</td>
      <td>...</td>
      <td>0.0121</td>
      <td>0.0036</td>
      <td>0.0150</td>
      <td>0.0085</td>
      <td>0.0073</td>
      <td>0.0050</td>
      <td>0.0044</td>
      <td>0.0040</td>
      <td>0.0117</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0762</td>
      <td>0.0666</td>
      <td>0.0481</td>
      <td>0.0394</td>
      <td>0.0590</td>
      <td>0.0649</td>
      <td>0.1209</td>
      <td>0.2467</td>
      <td>0.3564</td>
      <td>0.4459</td>
      <td>...</td>
      <td>0.0031</td>
      <td>0.0054</td>
      <td>0.0105</td>
      <td>0.0110</td>
      <td>0.0015</td>
      <td>0.0072</td>
      <td>0.0048</td>
      <td>0.0107</td>
      <td>0.0094</td>
      <td>R</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>




```python
#NUMBER OF ROWS AND COLUMS
sonar_data.shape
```




    (208, 61)




```python
#DESCRIBE THE DATA
sonar_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>...</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.029164</td>
      <td>0.038437</td>
      <td>0.043832</td>
      <td>0.053892</td>
      <td>0.075202</td>
      <td>0.104570</td>
      <td>0.121747</td>
      <td>0.134799</td>
      <td>0.178003</td>
      <td>0.208259</td>
      <td>...</td>
      <td>0.016069</td>
      <td>0.013420</td>
      <td>0.010709</td>
      <td>0.010941</td>
      <td>0.009290</td>
      <td>0.008222</td>
      <td>0.007820</td>
      <td>0.007949</td>
      <td>0.007941</td>
      <td>0.006507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022991</td>
      <td>0.032960</td>
      <td>0.038428</td>
      <td>0.046528</td>
      <td>0.055552</td>
      <td>0.059105</td>
      <td>0.061788</td>
      <td>0.085152</td>
      <td>0.118387</td>
      <td>0.134416</td>
      <td>...</td>
      <td>0.012008</td>
      <td>0.009634</td>
      <td>0.007060</td>
      <td>0.007301</td>
      <td>0.007088</td>
      <td>0.005736</td>
      <td>0.005785</td>
      <td>0.006470</td>
      <td>0.006181</td>
      <td>0.005031</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001500</td>
      <td>0.000600</td>
      <td>0.001500</td>
      <td>0.005800</td>
      <td>0.006700</td>
      <td>0.010200</td>
      <td>0.003300</td>
      <td>0.005500</td>
      <td>0.007500</td>
      <td>0.011300</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000800</td>
      <td>0.000500</td>
      <td>0.001000</td>
      <td>0.000600</td>
      <td>0.000400</td>
      <td>0.000300</td>
      <td>0.000300</td>
      <td>0.000100</td>
      <td>0.000600</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.013350</td>
      <td>0.016450</td>
      <td>0.018950</td>
      <td>0.024375</td>
      <td>0.038050</td>
      <td>0.067025</td>
      <td>0.080900</td>
      <td>0.080425</td>
      <td>0.097025</td>
      <td>0.111275</td>
      <td>...</td>
      <td>0.008425</td>
      <td>0.007275</td>
      <td>0.005075</td>
      <td>0.005375</td>
      <td>0.004150</td>
      <td>0.004400</td>
      <td>0.003700</td>
      <td>0.003600</td>
      <td>0.003675</td>
      <td>0.003100</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.022800</td>
      <td>0.030800</td>
      <td>0.034300</td>
      <td>0.044050</td>
      <td>0.062500</td>
      <td>0.092150</td>
      <td>0.106950</td>
      <td>0.112100</td>
      <td>0.152250</td>
      <td>0.182400</td>
      <td>...</td>
      <td>0.013900</td>
      <td>0.011400</td>
      <td>0.009550</td>
      <td>0.009300</td>
      <td>0.007500</td>
      <td>0.006850</td>
      <td>0.005950</td>
      <td>0.005800</td>
      <td>0.006400</td>
      <td>0.005300</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.035550</td>
      <td>0.047950</td>
      <td>0.057950</td>
      <td>0.064500</td>
      <td>0.100275</td>
      <td>0.134125</td>
      <td>0.154000</td>
      <td>0.169600</td>
      <td>0.233425</td>
      <td>0.268700</td>
      <td>...</td>
      <td>0.020825</td>
      <td>0.016725</td>
      <td>0.014900</td>
      <td>0.014500</td>
      <td>0.012100</td>
      <td>0.010575</td>
      <td>0.010425</td>
      <td>0.010350</td>
      <td>0.010325</td>
      <td>0.008525</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.137100</td>
      <td>0.233900</td>
      <td>0.305900</td>
      <td>0.426400</td>
      <td>0.401000</td>
      <td>0.382300</td>
      <td>0.372900</td>
      <td>0.459000</td>
      <td>0.682800</td>
      <td>0.710600</td>
      <td>...</td>
      <td>0.100400</td>
      <td>0.070900</td>
      <td>0.039000</td>
      <td>0.035200</td>
      <td>0.044700</td>
      <td>0.039400</td>
      <td>0.035500</td>
      <td>0.044000</td>
      <td>0.036400</td>
      <td>0.043900</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>




```python
sonar_data[60].value_counts()
```




    60
    M    111
    R     97
    Name: count, dtype: int64



M --> MINE
R --> ROCK


```python
sonar_data.groupby(60).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
    <tr>
      <th>60</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <td>0.034989</td>
      <td>0.045544</td>
      <td>0.050720</td>
      <td>0.064768</td>
      <td>0.086715</td>
      <td>0.111864</td>
      <td>0.128359</td>
      <td>0.149832</td>
      <td>0.213492</td>
      <td>0.251022</td>
      <td>...</td>
      <td>0.019352</td>
      <td>0.016014</td>
      <td>0.011643</td>
      <td>0.012185</td>
      <td>0.009923</td>
      <td>0.008914</td>
      <td>0.007825</td>
      <td>0.009060</td>
      <td>0.008695</td>
      <td>0.006930</td>
    </tr>
    <tr>
      <th>R</th>
      <td>0.022498</td>
      <td>0.030303</td>
      <td>0.035951</td>
      <td>0.041447</td>
      <td>0.062028</td>
      <td>0.096224</td>
      <td>0.114180</td>
      <td>0.117596</td>
      <td>0.137392</td>
      <td>0.159325</td>
      <td>...</td>
      <td>0.012311</td>
      <td>0.010453</td>
      <td>0.009640</td>
      <td>0.009518</td>
      <td>0.008567</td>
      <td>0.007430</td>
      <td>0.007814</td>
      <td>0.006677</td>
      <td>0.007078</td>
      <td>0.006024</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 60 columns</p>
</div>




```python
#SEPARATING DATA AND LABELS
x = sonar_data.drop(columns=60, axis= 1)
y = sonar_data[60]
```


```python
print(x)
print(y)
```

             0       1       2       3       4       5       6       7       8   \
    0    0.0200  0.0371  0.0428  0.0207  0.0954  0.0986  0.1539  0.1601  0.3109   
    1    0.0453  0.0523  0.0843  0.0689  0.1183  0.2583  0.2156  0.3481  0.3337   
    2    0.0262  0.0582  0.1099  0.1083  0.0974  0.2280  0.2431  0.3771  0.5598   
    3    0.0100  0.0171  0.0623  0.0205  0.0205  0.0368  0.1098  0.1276  0.0598   
    4    0.0762  0.0666  0.0481  0.0394  0.0590  0.0649  0.1209  0.2467  0.3564   
    ..      ...     ...     ...     ...     ...     ...     ...     ...     ...   
    203  0.0187  0.0346  0.0168  0.0177  0.0393  0.1630  0.2028  0.1694  0.2328   
    204  0.0323  0.0101  0.0298  0.0564  0.0760  0.0958  0.0990  0.1018  0.1030   
    205  0.0522  0.0437  0.0180  0.0292  0.0351  0.1171  0.1257  0.1178  0.1258   
    206  0.0303  0.0353  0.0490  0.0608  0.0167  0.1354  0.1465  0.1123  0.1945   
    207  0.0260  0.0363  0.0136  0.0272  0.0214  0.0338  0.0655  0.1400  0.1843   
    
             9   ...      50      51      52      53      54      55      56  \
    0    0.2111  ...  0.0232  0.0027  0.0065  0.0159  0.0072  0.0167  0.0180   
    1    0.2872  ...  0.0125  0.0084  0.0089  0.0048  0.0094  0.0191  0.0140   
    2    0.6194  ...  0.0033  0.0232  0.0166  0.0095  0.0180  0.0244  0.0316   
    3    0.1264  ...  0.0241  0.0121  0.0036  0.0150  0.0085  0.0073  0.0050   
    4    0.4459  ...  0.0156  0.0031  0.0054  0.0105  0.0110  0.0015  0.0072   
    ..      ...  ...     ...     ...     ...     ...     ...     ...     ...   
    203  0.2684  ...  0.0203  0.0116  0.0098  0.0199  0.0033  0.0101  0.0065   
    204  0.2154  ...  0.0051  0.0061  0.0093  0.0135  0.0063  0.0063  0.0034   
    205  0.2529  ...  0.0155  0.0160  0.0029  0.0051  0.0062  0.0089  0.0140   
    206  0.2354  ...  0.0042  0.0086  0.0046  0.0126  0.0036  0.0035  0.0034   
    207  0.2354  ...  0.0181  0.0146  0.0129  0.0047  0.0039  0.0061  0.0040   
    
             57      58      59  
    0    0.0084  0.0090  0.0032  
    1    0.0049  0.0052  0.0044  
    2    0.0164  0.0095  0.0078  
    3    0.0044  0.0040  0.0117  
    4    0.0048  0.0107  0.0094  
    ..      ...     ...     ...  
    203  0.0115  0.0193  0.0157  
    204  0.0032  0.0062  0.0067  
    205  0.0138  0.0077  0.0031  
    206  0.0079  0.0036  0.0048  
    207  0.0036  0.0061  0.0115  
    
    [208 rows x 60 columns]
    0      R
    1      R
    2      R
    3      R
    4      R
          ..
    203    M
    204    M
    205    M
    206    M
    207    M
    Name: 60, Length: 208, dtype: object
    

## TRAINING & TEST DATA


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, stratify = y,random_state = 1)
```


```python
print(x.shape, x_train.shape, x_test.shape)
```

    (208, 60) (187, 60) (21, 60)
    

MODEL TRAINING --> LOGISTIC REGRESSION


```python
model = LogisticRegression()
```


```python
#traning the Logistic Regression MODEL with traning data
model.fit(x_train, y_train)

```







MODEL EVALUATION


```python
#accuracy on traning data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
```


```python
print('Accuracy an training data :',training_data_accuracy)
```

    Accuracy an training data : 0.8342245989304813
    


```python
#accuracy on traning data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
```


```python
print('Accuracy an training data :',test_data_accuracy)
```

    Accuracy an training data : 0.7619047619047619
    

Making a Predictive System


```python
input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.099,0.1201,0.1833,0.2105,0.3039,0.2988,0.425,0.6343,0.8198,1,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.589,0.2872,0.2043,0.5782,0.5389,0.375,0.3411,0.5067,0.558,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.265,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)
#changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print('The object is a rock')
else:
    print('The object is a mine')




```

    ['R']
    The object is a rock
    
