#!/usr/bin/env python
# coding: utf-8

# ## Python code on a project on breast cancer wisconsin data

# In[ ]:


# Import Library
import pandas as pd
import numpy as np
from math import exp


# In[ ]:


# creating the list of names of hte columns to determine the correct label
names = ["ID",
         "Clump Thickness",
         "Uniformity of Cell Size",
         "Uniformity of Cell Shape",
         "Marginal Adhesion",
         "Single Epithelial Cell Size",
         "Bare Nuclei",
         "Bland Chromatin",
         "Normal Nucleoli",
         "Mitoses",
         "Class"]
data = pd.read_csv("breast-cancer-wisconsin.data", names=names, index_col=0) # extract the data

target = data.Class.apply(lambda x: 0 if x == 2 else 1) # applyig o and 1 for the benign and severe
data["Bare Nuclei"] = data["Bare Nuclei"].replace(["?"], -1).apply(lambda x: int(x) if x != -1 else 0) 
parameters = data.loc[:, data.columns != 'Class']
n_params = (parameters - parameters.mean(skipna=True)) / parameters.std(skipna=True)
n_params["Dummy const"] = 1


# In[2]:


# Applying Newton's Method 

def Newton(param, target):# define a function with arguments param and target
    N = param.shape[1] # column size of the param
    theta_p = np.full(N, 1)
    theta = np.full(N, 0)
    while np.linalg.norm(theta - theta_p) > 0.00001:
        mu = np.array([1 / (1 + exp(np.dot(theta, x))) for x in param])
        S = np.diag(mu * (1-mu))
        theta_p = theta
        H = np.dot(np.dot(param.T, S), param)
        theta = theta + np.dot(np.dot(np.linalg.inv(H), param.T), (mu - target))
    return theta

#pred = Newton(n_params.to_numpy(), target.to_numpy())

def Predict(model, data):
    return 0 if np.dot(model, data) > 0 else 1


# In[3]:


# Checking Validation
def Validate(param, target):
    sample_size = int(0.8 * len(target))
    samples = np.array(range(len(target)))
    np.random.shuffle(samples)
    theta = Newton(param.iloc[samples[:sample_size]].to_numpy(), target.iloc[samples[:sample_size]].to_numpy())
    success = 0
    fail = 0
    for i in range(sample_size, len(target)):
        if Predict(theta, param.iloc[i]) == target.iloc[i]:
            success += 1
        else:
            fail += 1
    return success / (fail + success)

results = [Validate(n_params, target) for _ in range(20)]
print("Average success ratio is ", sum(results) / len(results))


# 
