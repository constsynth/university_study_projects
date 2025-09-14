# Оптимизация портфеля в модели Хестона


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
```


```python
# Parameters
# simulation dependent
S0 = 100.0             # asset price
T = 1.0                # time in years
r = 0.02               # risk-free rate
N = 252                # number of time steps in simulation
M = 1000               # number of simulations

# Heston dependent parameters
kappa = 3              # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.20**2        # long-term mean of variance under risk-neutral dynamics
v0 = 0.25**2           # initial variance under risk-neutral dynamics
rho = 0.7              # correlation between returns and variances under risk-neutral dynamics
sigma = 0.6            # volatility of volatility

theta, v0
```




    (0.04000000000000001, 0.0625)




```python
def heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, N, M):
    """
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - T     : time of simulation
     - N     : number of time steps
     - M     : number of scenarios / simulations
    
    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = T/N
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])

    # arrays for storing prices and variances
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)

    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N,M))

    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)
    
    return S, v
```


```python
rho_p = 0.98
rho_n = -0.98
S_p,v_p = heston_model_sim(S0, v0, rho_p, kappa, theta, sigma,T, N, M)
S_n,v_n = heston_model_sim(S0, v0, rho_n, kappa, theta, sigma,T, N, M)
```


```python
fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12,5))
time = np.linspace(0,T,N+1)
ax1.plot(time,S_p)
ax1.set_title('Heston Model Asset Prices')
ax1.set_xlabel('Time')
ax1.set_ylabel('Asset Prices')

ax2.plot(time,v_p)
ax2.set_title('Heston Model Variance Process')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')

plt.show()
```


    
![png](course_project_files/course_project_5_0.png)
    



```python
# simulate gbm process at time T
gbm = S0*np.exp( (r - theta**2/2)*T + np.sqrt(theta)*np.sqrt(T)*np.random.normal(0,1,M) )

fig, ax = plt.subplots()

ax = sns.kdeplot(S_p[-1], label=r"$\rho= 0.98$", ax=ax)
ax = sns.kdeplot(S_n[-1], label=r"$\rho= -0.98$", ax=ax)
ax = sns.kdeplot(gbm, label="GBM", ax=ax)

plt.title(r'Asset Price Density under Heston Model')
plt.xlim([20, 180])
plt.xlabel('$S_T$')
plt.ylabel('Density')
plt.legend()
plt.show()
```


    
![png](course_project_files/course_project_6_0.png)
    



```python
rho = -0.7
S,v = heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, N, M)

# Set strikes and complete MC option price for different strikes
K = np.arange(20,180,2)

puts = np.array([np.exp(-r*T)*np.mean(np.maximum(k-S,0)) for k in K])
calls = np.array([np.exp(-r*T)*np.mean(np.maximum(S-k,0)) for k in K])

put_ivs = implied_vol(puts, S0, K, T, r, flag='p', q=0, return_as='numpy', on_error='ignore')
call_ivs = implied_vol(calls, S0, K, T, r, flag='c', q=0, return_as='numpy')

plt.plot(K, call_ivs, label=r'IV calls')
plt.plot(K, put_ivs, label=r'IV puts')

plt.ylabel('Implied Volatility')
plt.xlabel('Strike')

plt.title('Implied Volatility Smile from Heston Model')
plt.legend()
plt.show()
```

    /home/eugene/miniconda3/lib/python3.12/site-packages/py_vollib_vectorized/implied_volatility.py:75: UserWarning: Found Below Intrinsic contracts at index [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
      below_intrinsic, above_max_price = _check_below_and_above_intrinsic(K, F, flag, undiscounted_option_price, on_error)



    
![png](course_project_files/course_project_7_1.png)
    


# Оптимизация портфеля в модели GARCH(1, 1)


```python
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
```

#### Сгенерируем данные для дальнейшего обучения модели GARCH


```python
# Создадим датасет
n = 1000
omega = 0.5

alpha_1 = 0.1
alpha_2 = 0.2

beta_1 = 0.3
beta_2 = 0.4

test_size = int(n*0.25)

series = [gauss(0,1), gauss(0,1)]
vols = [1, 1]

for _ in range(n):
    new_vol = np.sqrt(omega + alpha_1*series[-1]**2 + alpha_2*series[-2]**2 + beta_1*vols[-1]**2 + beta_2*vols[-2]**2)
    new_val = gauss(0,1) * new_vol
    
    vols.append(new_vol)
    series.append(new_val)
```


```python
plt.figure(figsize=(10,4))
plt.plot(series)
plt.title('Симулированные данные для модели GARCH(1, 1)', fontsize=20)
```




    Text(0.5, 1.0, 'Симулированные данные для модели GARCH(1, 1)')




    
![png](course_project_files/course_project_12_1.png)
    



```python
plt.figure(figsize=(10,4))
plt.plot(vols)
plt.title('Стандартные отклонения симулированных данных', fontsize=20)
```




    Text(0.5, 1.0, 'Стандартные отклонения симулированных данных')




    
![png](course_project_files/course_project_13_1.png)
    


#### Построим график PACF (частичная автокорреляция)


```python
plot_pacf(np.array(series)**2)
plt.title('График частичной автокорреляционной функции для симулированных данных')
plt.show()
```


    
![png](course_project_files/course_project_15_0.png)
    


#### Обучим модель GARCH(1, 1)


```python
train, test = series[:-test_size], series[-test_size:]
```


```python
model = arch_model(train, p=1, q=1)
```


```python

model_fit = model.fit()
```

    Iteration:      1,   Func. Count:      6,   Neg. LLF: 3673.594349356084
    Iteration:      2,   Func. Count:     13,   Neg. LLF: 2953.4004907499
    Iteration:      3,   Func. Count:     19,   Neg. LLF: 234405532.81268573
    Iteration:      4,   Func. Count:     25,   Neg. LLF: 1988.0519529618516
    Iteration:      5,   Func. Count:     31,   Neg. LLF: 1978.703593131595
    Iteration:      6,   Func. Count:     37,   Neg. LLF: 4832.959794748294
    Iteration:      7,   Func. Count:     44,   Neg. LLF: 1978.505051901491
    Iteration:      8,   Func. Count:     50,   Neg. LLF: 1978.0981586196604
    Iteration:      9,   Func. Count:     56,   Neg. LLF: 1978.090902104047
    Iteration:     10,   Func. Count:     61,   Neg. LLF: 1978.0909014159736
    Optimization terminated successfully    (Exit mode 0)
                Current function value: 1978.0909014159736
                Iterations: 10
                Function evaluations: 61
                Gradient evaluations: 10



```python
model_fit.summary()
```




<table class="simpletable">
<caption>Constant Mean - GARCH Model Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>y</td>         <th>  R-squared:         </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Mean Model:</th>       <td>Constant Mean</td>   <th>  Adj. R-squared:    </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Vol Model:</th>            <td>GARCH</td>       <th>  Log-Likelihood:    </th> <td>  -1978.09</td>
</tr>
<tr>
  <th>Distribution:</th>        <td>Normal</td>       <th>  AIC:               </th> <td>   3964.18</td>
</tr>
<tr>
  <th>Method:</th>        <td>Maximum Likelihood</td> <th>  BIC:               </th> <td>   3982.67</td>
</tr>
<tr>
  <th></th>                        <td></td>          <th>  No. Observations:  </th>     <td>752</td>   
</tr>
<tr>
  <th>Date:</th>           <td>Sat, Jun 14 2025</td>  <th>  Df Residuals:      </th>     <td>751</td>   
</tr>
<tr>
  <th>Time:</th>               <td>21:02:44</td>      <th>  Df Model:          </th>      <td>1</td>    
</tr>
</table>
<table class="simpletable">
<caption>Mean Model</caption>
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>     <th>95.0% Conf. Int.</th>  
</tr>
<tr>
  <th>mu</th> <td>    0.1515</td> <td>9.859e-02</td> <td>    1.537</td> <td>    0.124</td> <td>[-4.168e-02,  0.345]</td>
</tr>
</table>
<table class="simpletable">
<caption>Volatility Model</caption>
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>    <th>95.0% Conf. Int.</th>  
</tr>
<tr>
  <th>omega</th>    <td>    0.2486</td> <td>9.099e-02</td> <td>    2.732</td> <td>6.289e-03</td> <td>[7.027e-02,  0.427]</td>
</tr>
<tr>
  <th>alpha[1]</th> <td>    0.1354</td> <td>2.207e-02</td> <td>    6.133</td> <td>8.625e-10</td> <td>[9.211e-02,  0.179]</td>
</tr>
<tr>
  <th>beta[1]</th>  <td>    0.8536</td> <td>1.871e-02</td> <td>   45.611</td>   <td>0.000</td>    <td>[  0.817,  0.890]</td> 
</tr>
</table><br/><br/>Covariance estimator: robust



#### Сделаем предсказания для тестовой выборки и сравним графики волатильности


```python
predictions = model_fit.forecast(horizon=test_size)
```


```python
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions.variance.values[-1, :]))
plt.title('Предсказание волатильности для тестовой выборки', fontsize=20)
plt.legend(['Истинная волатильность', 'Предсказанная волатильность'], fontsize=16)
```




    <matplotlib.legend.Legend at 0x7603c52d8ce0>




    
![png](course_project_files/course_project_23_1.png)
    



```python
rolling_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=1, q=1)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
```


```python
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Предсказание волатильности для тестовой выборки скользящим окном', fontsize=20)
plt.legend(['Истинная волатильность', 'Предсказанная волатильность'], fontsize=16)
```




    <matplotlib.legend.Legend at 0x7603c993e0f0>




    
![png](course_project_files/course_project_25_1.png)
    

