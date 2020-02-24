#!/usr/bin/env python
# coding: utf-8

# $$\large \color{green}{\textbf{The Value-At-Risk Measurements with Filtered Historical Simulation }}$$ 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# 
# This computer program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# $$\text{1. Issue}$$
# 
# One of the most frequently used aspects of the volatility models is to measure the Value-At-Risk (VaR). This project attempts to use the GARCH model to measure the VaR.
# 
# $$\text{2. Methodology}$$
# 
# The GARCH model specification
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Volatility equation:}$$
# $$\sigma^{2}_{t}= \omega + \alpha \epsilon^{2}_{t} + \beta\sigma^{2}_{t-1}$$
# 
# $$\text{Volatility equation:}$$
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 
# we use the model to estimate the VaR
# 
# 
# Value-at-Risk (VaR) forecasts from GARCH models depend on the conditional mean, the conditional volatility and the quantile of the standardized residuals,
# 
# $$\text{VaR}_{t+1|t}=\mu_{t+1} -\sigma_{t+1|t}q_{\alpha} $$
# 
# 
# where $q_{\alpha}$ is the $\alpha$ quantile of the standardized residuals, e.g., 5%. It is worth noting that there are a number of methods to calculate this qualtile, such as the parametric (or the variance–covariance approach), the Historical Simulation, and the Monte Carlo simulation. For example, my previous project used the Parametric method which under skewed and fat-tail distributions also provides promising results especially when the assumption that standardised returns are independent and identically distributed is set aside and when time variations are considered in conditional high-order moments. However, From a practical perspective, empirical literature shows that approaches based on the Extreme Value Theory and the Filtered Historical Simulation are the best methods for forecasting VaR.Thus, this project uses the Filtered Historical Simulation to calculate the VaR.
# 
# 
# $$\text{3. Dataset}$$ 
# 
# One can download the dataset used to replicate my project at my Repositories on the Github site below
# 
# https://github.com/phuongvnguyen/The-VaR-Caculation-with-the-Filtered-Historical-Simulation
# 
# 
# # Preparing Problem
# 
# ##  Loading Libraries

# In[1]:


import warnings
import itertools
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from arch import arch_model
from arch.univariate import GARCH


# ## Defining some varibales for printing the result

# In[2]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[3]:


data = pd.read_excel("mydata.xlsx")


# # Data Exploration and Preration
# 
# ## Data exploration

# In[4]:


data.head(5)


# ## Computing returns
# ### Picking up the close prices

# In[5]:


closePrice = data[['DATE','CLOSE']]
closePrice.head(5)


# ### Computing the daily returns

# In[6]:


closePrice['Return'] = closePrice['CLOSE'].pct_change()
closePrice.head()


# In[7]:


daily_return=closePrice[['DATE','Return']]
daily_return.head()


# ### Reseting index

# In[8]:


daily_return =daily_return.set_index('DATE')
daily_return.head()


# In[9]:


daily_return = 100 * daily_return.dropna()
daily_return.head()


# In[10]:


daily_return.index


# ### Plotting returns

# In[11]:


sns.set()
fig=plt.figure(figsize=(12,7))
plt.plot(daily_return.Return['2007':'2018'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Log Daily Returns', fontsize=15,fontweight='bold'
             ,color='b')
plt.title('20/09/2007- 28/12/2018',fontsize=13,fontweight='bold',
          color='b')
plt.ylabel('Return (%)',fontsize=10)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=10,fontweight='normal',color='k')


# # Modelling GARCH model
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Volatility equation:}$$
# $$\sigma^{2}_{t}= \omega + \alpha \epsilon^{2}_{t} + \beta\sigma^{2}_{t-1}$$
# 
# $$\text{Volatility equation:}$$
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 

# In[12]:


for row in daily_return.index: 
    print(row)


# In[13]:


#garch = arch_model(daily_return,mean='AR',lags=5,
 #                  vol='GARCH',dist='studentst',
  #              p=1, o=0, q=1)
garch = arch_model(daily_return,vol='Garch', p=1, o=0, q=1, dist='skewt')
results_garch = garch.fit(last_obs='2017-12-29', update_freq=1,disp='on')
print(results_garch.summary())


# # Estimating the VaR
# 
# we use the model to estimate the VaR
# 
# 
# Value-at-Risk (VaR) forecasts from GARCH models depend on the conditional mean, the conditional volatility and the quantile of the standardized residuals,
# 
# $$\text{VaR}_{t+1|t}=\mu_{t+1} -\sigma_{t+1|t}q_{\alpha} $$
# 
# 
# where $q_{\alpha}$ is the $\alpha$ quantile of the standardized residuals, e.g., 5%.
# 
# ## Computing the quantiles
# 
# The quantiles, $q_{\alpha}$, can be computed using the Filtered Historical Simulation below.
# 
# ### Computing the standardized residuals
# 
# It is worth noting that the standardized residuals computed by conditional volatility as follows.

# In[18]:


std_garch = (daily_return.Return['2007':'2018'] - results_garch.params['mu']) / results_garch.conditional_volatility
std_garch = std_garch.dropna()
std_garch.head(5)


# ### Computing the Quantiles
# 
# At the probabilities of 1% and 5%

# In[20]:


FHS_quantiles_VaRgarch = std_garch.quantile([.01, .05])
print(Bold+'The quantiles at the probabilities of 1% and 5% are as follows'+End)
print(FHS_quantiles_VaRgarch)


# ## Computing the conditional mean and volatilitie

# In[23]:


FHS_forecasts_VaRgarch = results_garch.forecast(start='2018-01-03')
FHS_cond_mean_VaRgarch = FHS_forecasts_VaRgarch.mean['2018':]
FHS_cond_var_VaRgarch = FHS_forecasts_VaRgarch.variance['2018':]


# ## Computing the Value-At-Risk (VaR)

# In[ ]:


FHS_value_at_risk = -FHS_cond_mean.values - np.sqrt(cond_var).values * q.values[None, :]
value_at_risk = pd.DataFrame(
    value_at_risk, columns=['1%', '5%'], index=cond_var.index)


# In[26]:


FHS_value_at_risk = -FHS_cond_mean_VaRgarch.values - np.sqrt(FHS_cond_var_VaRgarch).values * FHS_quantiles_VaRgarch[None, :]

FHS_value_at_risk = pd.DataFrame(
    FHS_value_at_risk, columns=['1%', '5%'], index=FHS_cond_var_VaRgarch.index)

FHS_value_at_risk.head(5)


# In[27]:


FHS_value_at_risk = FHS_value_at_risk.dropna()
FHS_value_at_risk.head(5)


# # Visualizing the VaR vs actual values
# ## Picking actual data

# In[28]:


rets_2018= daily_return['2018':].copy()
rets_2018.name = 'Return'
rets_2018.head(5)


# ## Plotting

# In[29]:


fig=plt.figure(figsize=(12,5))
plt.plot(FHS_value_at_risk['1%'] ,LineWidth=2,
         linestyle='--',label='VaR returns at 1%')
plt.plot(FHS_value_at_risk['5%'] ,LineWidth=2,
         linestyle=':',label='VaR returns at 5%')
plt.plot(rets_2018['Return'] ,LineWidth=2,
         linestyle='-',label='Actual return')
plt.suptitle('The Daily GARCH-based Value-At-Risk (VaR) Measurements of the Vingroup stock', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('01/02/2018-28/12/2018',fontsize=10,
          fontweight='bold',color='b')
plt.autoscale(enable=True,axis='both',tight=True)
plt.legend()

