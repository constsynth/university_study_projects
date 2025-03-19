# Лабораторная работа 5 "Множественная регрессия"
# Вариант №3


```python
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

## Считываем набор данных для третьего варианта


```python
df = pd.read_excel('v3_table.xlsx')
```


```python
df.head()
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
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48</td>
      <td>44</td>
      <td>47</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>35</td>
      <td>56</td>
      <td>35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>39</td>
      <td>54</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61</td>
      <td>43</td>
      <td>62</td>
      <td>39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>36</td>
      <td>56</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Рассчитать параметры линейного уравнения множественной регрессии с полным перечнем факторов. 


```python
X = df[['x1', 'x2', 'x3']]
X = sm.add_constant(X)  # Добавляем константу
y = df['y']
```


```python
model = sm.OLS(y, X).fit() # Обучаем модель множественной регрессии
```


```python
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.946
    Model:                            OLS   Adj. R-squared:                  0.930
    Method:                 Least Squares   F-statistic:                     58.89
    Date:                Wed, 19 Mar 2025   Prob (F-statistic):           1.17e-06
    Time:                        22:07:14   Log-Likelihood:                -28.102
    No. Observations:                  14   AIC:                             64.20
    Df Residuals:                      10   BIC:                             66.76
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          3.3248      5.909      0.563      0.586      -9.841      16.490
    x1             0.0010      0.965      0.001      0.999      -2.149       2.151
    x2             0.1356      0.151      0.898      0.390      -0.201       0.472
    x3             0.5236      0.962      0.544      0.598      -1.619       2.666
    ==============================================================================
    Omnibus:                        1.198   Durbin-Watson:                   2.434
    Prob(Omnibus):                  0.549   Jarque-Bera (JB):                0.515
    Skew:                          -0.467   Prob(JB):                        0.773
    Kurtosis:                       2.898   Cond. No.                         899.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    /home/eugene/miniconda3/lib/python3.12/site-packages/scipy/stats/_stats_py.py:1971: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=14
      k, _ = kurtosistest(a, axis)


## 2. Оценить значимость уравнения в целом, используя значение множественного коэффициента корреляции и общего F-критерия Фишера.


```python
print(f"Множественный коэффициент корреляции: {np.sqrt(model.rsquared)}")
print(f"F-статистика: {model.fvalue}, p-value: {model.f_pvalue}")
```

    Множественный коэффициент корреляции: 0.9728471690773998
    F-статистика: 58.89242342360955, p-value: 1.1671216997748412e-06


Выводы из пункта:
- 97.2 % изменчивости целевой переменной `y` объяснено построенным уравнением множественной регрессии;
- Посколкьу p-value для рассчитанной F-статистики существенно ниже уровня значимости 0.05, мы можем сказать, что построенное уравнение множественной регрессии является стат. значимым 

## 3. Оценить статистическую значимость параметров регрессионной модели с помощью t-критерия.


```python
print(model.tvalues)
```

    const    0.562693
    x1       0.001068
    x2       0.898166
    x3       0.544480
    dtype: float64



```python
alpha = 0.05
n = len(df)  # Количество примеров в датасете
k = len(model.params) - 1  # Количество независимых переменных (не считая )

# Степени свободы
degrees_of_freedom = n - k - 1

# Критическое значение t-статистики (двусторонний тест)
t_critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

print(f"Критическое значение t-статистики (α = {alpha}, degrees_of_freedom = {degrees_of_freedom}): {t_critical:.4f}")
```

    Критическое значение t-статистики (α = 0.05, degrees_of_freedom = 10): 2.2281



```python
for ind_var_t_stat in model.tvalues.tolist():
    if abs(ind_var_t_stat) > abs(t_critical):
        print(True)
    else:
        print(False)
```

    False
    False
    False
    False


Выводы из пункта:
- Поскольку значения t-статистики для каждой из нецелевых переменных не превышают по модулю t-stat.crit., данное уравнение не может быть использовано для прогнозирования

## Исследовать коллинеарность между факторами. При наличии мультиколлинеарности исключить какой-либо фактор из уравнения регрессии.


```python
corr_matrix = df[['x1', 'x2', 'x3']].corr()
```


```python
corr_matrix
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
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x1</th>
      <td>1.000000</td>
      <td>0.361976</td>
      <td>0.999022</td>
    </tr>
    <tr>
      <th>x2</th>
      <td>0.361976</td>
      <td>1.000000</td>
      <td>0.373459</td>
    </tr>
    <tr>
      <th>x3</th>
      <td>0.999022</td>
      <td>0.373459</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Матрица корреляции")
plt.show()
```


    
![png](LR_5_files/LR_5_20_0.png)
    


Видим, что третья и первая независимые переменные очень сильно между собой коррелируют. Поскольку коэффициенты парной корреляции у этих двух факторов примерно одинаковые, можно исключить любой из них. Например, `x3`.

## 5. Построить новое уравнение множественной регрессии, провести все необходимые исследования, аналогичные проведенным выше.



```python
X = df[['x1', 'x2']]
X = sm.add_constant(X)  # Добавляем константу
y = df['y']
```


```python
model = sm.OLS(y, X).fit() # Обучаем модель множественной регрессии
```


```python
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.945
    Model:                            OLS   Adj. R-squared:                  0.935
    Method:                 Least Squares   F-statistic:                     94.22
    Date:                Wed, 19 Mar 2025   Prob (F-statistic):           1.20e-07
    Time:                        22:23:02   Log-Likelihood:                -28.306
    No. Observations:                  14   AIC:                             62.61
    Df Residuals:                      11   BIC:                             64.53
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.0509      5.249      0.391      0.703      -9.503      13.605
    x1             0.5259      0.043     12.340      0.000       0.432       0.620
    x2             0.1592      0.140      1.138      0.279      -0.149       0.467
    ==============================================================================
    Omnibus:                        0.936   Durbin-Watson:                   2.199
    Prob(Omnibus):                  0.626   Jarque-Bera (JB):                0.155
    Skew:                          -0.252   Prob(JB):                        0.925
    Kurtosis:                       3.103   Cond. No.                         641.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    /home/eugene/miniconda3/lib/python3.12/site-packages/scipy/stats/_stats_py.py:1971: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=14
      k, _ = kurtosistest(a, axis)


#### Оценить значимость уравнения в целом, используя значение множественного коэффициента корреляции и общего F-критерия Фишера.


```python
print(f"Множественный коэффициент корреляции: {np.sqrt(model.rsquared)}")
print(f"F-статистика: {model.fvalue}, p-value: {model.f_pvalue}")
```

    Множественный коэффициент корреляции: 0.9720306255611043
    F-статистика: 94.21632885275885, p-value: 1.1988925065427086e-07


Выводы из пункта:
- 97.2 % изменчивости целевой переменной `y` объяснено построенным уравнением множественной регрессии;
- Посколкьу p-value для рассчитанной F-статистики существенно ниже уровня значимости 0.05, мы можем сказать, что построенное уравнение множественной регрессии является стат. значимым 

#### Оценить статистическую значимость параметров регрессионной модели с помощью t-критерия.


```python
print(model.tvalues)
```

    const     0.390698
    x1       12.340250
    x2        1.137959
    dtype: float64



```python
alpha = 0.05
n = len(df)  # Количество примеров в датасете
k = len(model.params) - 1  # Количество независимых переменных (не считая )

# Степени свободы
degrees_of_freedom = n - k - 1

# Критическое значение t-статистики (двусторонний тест)
t_critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

print(f"Критическое значение t-статистики (α = {alpha}, degrees_of_freedom = {degrees_of_freedom}): {t_critical:.4f}")
```

    Критическое значение t-статистики (α = 0.05, degrees_of_freedom = 11): 2.2010



```python
for ind_var_t_stat in model.tvalues.tolist():
    if abs(ind_var_t_stat) > abs(t_critical):
        print(True)
    else:
        print(False)
```

    False
    True
    False


Выводы из пункта:
- Независимая переменная `x1` имеет значение t-статистики существенно больше, чем t-stat.crit., следовательно, построенное уравнение множественной регрессии может использоваться для прогнозирования 

## 6. На основании результатов п. 5 найти
* средние коэффициенты эластичности фактора y от независимых
факторов;
* прогнозное значение результата при значении важнейшей
объясняющей переменной, равном максимальному наблюденному значению,
увеличенному на 10 %, и при значении второй объясняющей переменной,
равном минимальному наблюденному значению, уменьшенному на 15%.
* Интервальное предсказание значения y с надежностью 0,95.


```python
# а) Средние коэффициенты эластичности
elasticities = model.params[1:] * X.mean() / y.mean()
print("Средние коэффициенты эластичности:")
print(elasticities)
```

    Средние коэффициенты эластичности:
    const         NaN
    x1       0.767090
    x2       0.176164
    dtype: float64



```python
# б) Прогнозное значение
x1_max = df['x1'].max() * 1.10
x2_min = df['x2'].min() * 0.85
new_data = pd.DataFrame({'const': [1], 'x1': [x1_max], 'x2': [x2_min]})
prediction = model.predict(new_data)
print(f"Прогнозное значение y: {prediction[0]}")
```

    Прогнозное значение y: 50.928134031511775



```python
# в) Интервальное предсказание
prediction = model.get_prediction(new_data)
prediction_summary = prediction.summary_frame(alpha=0.05)
print(f"Интервальное предсказание y: [{prediction_summary.mean_ci_lower.tolist()[0]}, {prediction_summary.mean_ci_upper.tolist()[0]}]")
```

    Интервальное предсказание y: [45.04906833708088, 56.80719972594267]

