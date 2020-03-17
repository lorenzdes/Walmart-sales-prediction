#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import zscore

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


# In[2]:


train = pd.read_csv('\\Users\\Lorenzo de Sario\\Desktop\\walmart\\train.csv')
features = pd.read_csv('\\Users\\Lorenzo de Sario\\Desktop\\walmart\\features.csv')
test = pd.read_csv('\\Users\\Lorenzo de Sario\\Desktop\\walmart\\test.csv')


# In[3]:


features.info()


# In[4]:


features = features.iloc[:,[0,1,3,9,11]]
features


# In[5]:


data = pd.merge(train, features)
data


# In[6]:


data['Date'] = pd.to_datetime(data['Date']).dt.year


# In[7]:


data.plot(kind = 'box', subplots = 1, figsize = (20,16))
plt.show()


# In[8]:


plt.scatter(data['Weekly_Sales'], data['CPI'])


# In[9]:


features = ['IsHoliday', 'Fuel_Price', 'CPI']
x = data[features]

y = data['Weekly_Sales']


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state = 1)


# In[18]:


regr = LinearRegression()
regr.fit(x_train, y_train)
sales_prediction = regr.predict(x_test)
print('predictionssssssss\n', sales_prediction)
#predictionssssssss


# In[19]:


print('we have a variation from the real value of about',mean_absolute_error(y_test, sales_prediction))
#we have a variation from the real value of about


# In[ ]:


print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f' %mean_squared_error(y_test, sales_prediction))


# In[ ]:


start_time = time()
model_list = [LinearRegression()]
Score = []
for i in model_list:
    i.fit(x_train,y_train)
    y_pred = i.predict(x_test)
    score = r2_score(y_test,y_pred)
    Score.append(score)
print(pd.DataFrame(zip(model_list,Score), columns = ['Model Use d', 'R2-Score']))
end_time = time()
print(round(end_time-start_time,2), 'sec')


# In[ ]:




