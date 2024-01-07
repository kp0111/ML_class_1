#!/usr/bin/env python
# coding: utf-8

# In[1]:


#linear regression and ridge and lasso 


# In[3]:


from sklearn.datasets import load_boston


# In[4]:


import numpy as np


# In[5]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df=load_boston()


# In[13]:


df


# In[7]:


type(df)


# In[14]:


dataset = pd.DataFrame(df.data)
dataset.column = df.feature_names
print(dataset.head())


# In[15]:


dataset['Price'] = df.target


# In[17]:


dataset.head()


# In[19]:


## dividing dataset into dependent and independent fetaure
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]


# In[20]:


X.head()


# In[21]:


Y.head()


# In[26]:


# linear regression 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()
mse = cross_val_score(lin_reg,X,Y,scoring = 'neg_mean_squared_error',cv=5)
print(mse)


# In[27]:


mean_mse = np.mean(mse)
print(mean_mse)


# In[29]:


# ridge regression


# In[6]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge = Ridge()

params = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20]}
ridge_regressor = GridSearchCV(ridge,params,scoring = 'neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,Y)


# In[7]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston  # Example dataset, replace with your own data

# Load your dataset or define X and Y here
boston = load_boston()
X = boston.data
Y = boston.target

ridge = Ridge()
params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regressor = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X, Y)


# In[8]:


# to print the best parameter value
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[11]:


#lasso regression 
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston  # Example dataset, replace with your own data

# Load your dataset or define X and Y here
boston = load_boston()
X = boston.data
Y = boston.target

lasso = Lasso()
params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X, Y)


# In[14]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[19]:


# still we would be selecting the linear just for verification purpose if we add more value of alpha 
#lasso regression 
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston  # Example dataset, replace with your own data

# Load your dataset or define X and Y here
boston = load_boston()
X = boston.data
Y = boston.target

lasso = Lasso()
params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20,30,40,45,50,55,100]}
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X, Y)


# In[20]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# ## LOGISTIC REGRESSION

# In[31]:


from sklearn.linear_model import LogisticRegression
import pandas as pd


# In[32]:


from sklearn.datasets import load_breast_cancer


# In[29]:


# df=load_breast_cancer
# #independent features
# X = pd.DataFrame(df['data'],columns=df['featues_names'])


from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the breast cancer dataset
data = load_breast_cancer()

# Independent features
X = pd.DataFrame(data['data'], columns=data['feature_names'])


# In[30]:


X.head()


# In[38]:


# # dependented features from the datasets will be mentioned under y 
# Y = pd.DataFrame(data['target'],column=['target'])

Y = pd.DataFrame(data['target'], columns=['target'])


# In[39]:


Y


# In[42]:


Y['target'].value_counts()
# balnced data set 


# ## train _test _split

# In[44]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(
X,Y,test_size=0.33,random_state=42)


# In[68]:


# Parameters for grid search
params = [{'C': [1, 5, 10]}, {'max_iter': [100, 150]}]

# Creating the Logistic Regression model instance
model1 = LogisticRegression(C=100, max_iter=100)

# GridSearchCV
grid_search = GridSearchCV(model1, param_grid=params, scoring='f1', cv=5)
grid_search.fit(X_train, Y_train)


# In[70]:


best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)


# In[72]:


Y_pred = grid_search.predict(X_test)


# In[73]:


Y_pred


# In[74]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[75]:


confusion_matrix(Y_test,Y_pred)


# In[76]:


accuracy_score(Y_test,Y_pred)


# In[ ]:




