#!/usr/bin/env python
# coding: utf-8

# # Question One

# Data Collection:BP Equity Price Movement for the period January 2021 to January 2023

# In[1]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = yf.download('BP', start='2021-01-01', end='2023-01-01')


# In[4]:


data['Return'] = data['Close'].pct_change()


# Calculate annualized average return

# In[5]:


avg_return = data['Return'].mean() * 252
print(f'Annualized Average Return: {avg_return * 100:.2f}%')


# Calculate annualized standard deviation

# In[7]:


std_dev = data['Return'].std() * np.sqrt(252)
print(f'Annualized Standard Deviation: {std_dev * 100:.2f}%')


# # Plot equity price movement

# In[8]:


plt.figure(figsize=(10,6))
data['Close'].plot()
plt.title('BP Equity Price Movement (2021-01-01 to 2023-01-01)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# # Question Two

# Option Pricing models

# In[13]:


data_ac = data['Adj Close']
data_ac.head()


# In[14]:


S0 = data_ac[-1:]     # spot price = 33.395939
K = 34.93             # strike price
T = 1                 # maturity 1 year
r = 0.05              # risk free rate 
sig = std_dev         # diffusion coefficient or volatility
N = 3                 # number of periods or number of time steps  
payoff = "put"        # payoff 

print(S0)


# Binomial tree

# 1. Creating a Binomial Tree

# In[41]:


dT = float(T) / N                    # Delta t
u = np.exp(sig * np.sqrt(dT))        # up factor
d = 1.0 / u                          # down factor 

S1 = np.zeros((N + 1, N + 1))
S1[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S1[i, t] = S1[i, t-1] * u
        S1[i+1, t] = S1[i, t-1] * d
    z += 1


# In[16]:


print('The up factor u is ',(u))


# In[17]:


print('The down factor d is ',(d))


# In[18]:


print('The binomial tree presenting Polka Dot price over 3 time steps ', '\n', (S1))


# In[19]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability
print('Risk-neutral porbability is ', (p))


# Option value at each final nodes

# In[20]:


S_T = S1[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V


# Option value at earlier nodes

# In[21]:


# for European Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
print('The option values at earlier nodes ', '\n', (V))


# In[22]:


print('Binomial tree & Option price - European ' + payoff, str( V[0,0]))


# Monte Carlo Pricing Model

# In[23]:


def mcs_simulation_np(p):
    M = p
    I = p
    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sig ** 2 / 2) * dt + sig * np.sqrt(dt) * rn[t]) 
    return S


# In[24]:


S2 = mcs_simulation_np(1000)


# In[25]:


fig = plt.figure()
plt.plot(S2)
fig.suptitle('Monte Carlo Simulation: BP')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()


# In[26]:


S2 = np.transpose(S2)
S2


# In[27]:


n, bins, patches = plt.hist(x=S2[:,-1], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('S_T')
plt.ylabel('Frequency')
plt.title('Frequency distribution of the simulated end-of-preiod values')


# In[28]:


put = np.mean(np.maximum(K - S2[:,-1],0))
print('Monte Carlo Simulation & Option price - European put', str(put))


# # Question Three

# Greeks for Risk Management Purpose

# Black-scholes option value

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si


# In[44]:


def euro_option_bsm(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * std_dev ** 2) * T) / (std_dev * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * std_dev ** 2) * T) / (std_dev * np.sqrt(T))
    if payoff == "call":
        option_value = S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        option_value =  - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) + K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return option_value


# In[45]:


Bsm = euro_option_bsm(S0, K, T, r, 0, std_dev, 'put')            #q: dividend = 0
print('Black-scholes & Option price - European put', str(put))


# Delta

# In[46]:


def delta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        delta = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    elif payoff == "put":
        delta =  - np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)
    
    return delta


# In[48]:


delta(S0, K, T, r, 0, std_dev, 'put')


# In[49]:


S11 = np.linspace(3,15,11)
Delta_Call = np.zeros((len(S11),1))
Delta_Put = np.zeros((len(S11),1))
for i in range(len(S11)):
    Delta_Put [i] = delta(S11[i], K, T, r, 0, std_dev, 'put')


# In[50]:


fig = plt.figure()

plt.plot(S11, Delta_Put, '--')
plt.grid()
plt.xlabel('BP Price')
plt.ylabel('Delta')
plt.title('Delta')
plt.legend(['Delta for Put'])


# Gamma

# In[51]:


def gamma(S, K, T, r,  vol, payoff):
    
    d1 = (np.log(S / K) + (r  + 0.5 * std_dev ** 2) * T) / (vol * np.sqrt(T))

    gamma = si.norm.pdf(d1, 0.0, 1.0) / (vol *  np.sqrt(T) * S)

    
    return gamma


# In[52]:


gamma(33.39, 34.93, 1, 0.05, 0.347, 'put')


# In[53]:


S = np.linspace(50,150,11)
Gamma = np.zeros((len(S),1))
for i in range(len(S)):
    Gamma [i] = gamma(S[i], 34.93, 1, 0.05, 0.347, 'put')


# In[54]:


fig = plt.figure()
plt.plot(S, Gamma, '-')
plt.grid()
plt.xlabel('BP Price')
plt.ylabel('Gamma')
plt.title('Gamma')
plt.legend(['Gamma for Call and Put'])


# Theta

# In[60]:


def theta(S, K, T, r, std_dev, payoff):
    
    d1 = (np.log(S / K) + (r + 0.5 * std_dev ** 2) * T) / (std_dev * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * std_dev ** 2) * T) / (std_dev * np.sqrt(T))
    N_d1_prime=1/np.sqrt(2 * np.pi) * np.exp(-d1**2/2)
    
    if payoff == "call":
        theta = - S * N_d1_prime * std_dev / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        theta = - S * N_d1_prime * std_dev / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta


# In[61]:


theta(33.39, 34.93, 1, 0.05, 0.347, 'put')


# In[62]:


T = np.linspace(0.25,3,12)
Theta_Call = np.zeros((len(T),1))
Theta_Put = np.zeros((len(T),1))
for i in range(len(T)):
    Theta_Call [i] = theta(100, 100, T[i], 0.05, 0.25, 'call')
    Theta_Put [i] = theta(100, 100, T[i], 0.05, 0.25, 'put')


# In[65]:


fig = plt.figure()
plt.plot(T, Theta_Put, '-')
plt.grid()
plt.xlabel('Time to Expiry')
plt.ylabel('Theta')
plt.title('Theta')
plt.legend(['Theta for Put'])


# Rho

# In[66]:


def rho(S, K, T, r, std_dev, payoff):
    
    d1 = (np.log(S / K) + (r + 0.5 * std_dev ** 2) * T) / (std_dev * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * std_dev ** 2) * T) / (std_dev * np.sqrt(T))
    if payoff == "call":
        rho =  K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        rho = - K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return rho


# In[67]:


rho(33.39, 34.93, 1, 0.05, 0.347, 'put')


# In[68]:


r = np.linspace(0,0.1,11)
Rho_Call = np.zeros((len(r),1))
Rho_Put = np.zeros((len(r),1))
for i in range(len(r)):
    Rho_Put [i] = rho(100, 100, 1, r[i], 0.25, 'put')


# In[69]:


fig = plt.figure()
plt.plot(r, Rho_Call, '-')
plt.plot(r, Rho_Put, '-')
plt.grid()
plt.xlabel('Interest Rate')
plt.ylabel('Rho')
plt.title('Rho')
plt.legend(['Rho for Put'])


# Vega

# In[72]:


def vega(S, K, T, r, std_dev, payoff):
    
    d1 = (np.log(S / K) + (r + 0.5 * std_dev ** 2) * T) / (std_dev * np.sqrt(T))
    N_d1_prime=1/np.sqrt(2 * np.pi) * np.exp(-d1**2/2)
    vega = S * np.sqrt(T) * N_d1_prime
    
    return vega


# In[73]:


vega(33.39, 34.93, 1, 0.05, 0.347, 'put')


# In[74]:


vol = np.linspace(0.1,0.4,13)
Vega = np.zeros((len(vol),1))
for i in range(len(vol)):
    Vega [i] = vega(100, 100, 1, 0.05, vol[i], 'put')


# In[75]:


fig = plt.figure()
plt.plot(vol, Vega, '-')
plt.grid()
plt.xlabel('Volatility')
plt.ylabel('Vega')
plt.title('Vega')
plt.legend(['Vega for Put'])


# In[ ]:




