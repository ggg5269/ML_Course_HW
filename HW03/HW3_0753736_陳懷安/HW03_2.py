#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.io
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import scale

from PIL import Image
import os
import cv2
import operator


# In[2]:


def eigen(train_x):
    cov_train = np.cov(train_x.T)
    eig_val, eig_vec = np.linalg.eig(cov_train)
    return np.abs(eig_val), eig_vec

def PCA(train_x, n, eig_val_train, eig_vec_train):
    
    eig_pairs = [(np.abs(eig_val_train[i]), eig_vec_train[:,i]) for i in range(len(eig_val_train))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    tot = sum(eig_val_train[:n])
    var_exp = [(i / tot)*100 for i in sorted(eig_val_train[:n], reverse=True)]

    cum_var_exp = np.array(np.cumsum(var_exp), dtype = 'float32')

    pca_w = np.array([eig_pairs[i][1] for i in range(n)], dtype = 'float32')
    return train_x @ pca_w.T


# In[14]:


train_x = pd.read_csv('x_train.csv', header=None).values
train_t = pd.read_csv('t_train.csv', header=None).values.squeeze()


# In[15]:


eig_val, eig_vec = eigen(train_x)
train_x = PCA(train_x, 2, eig_val, eig_vec)
train_x = scale(train_x)


# # SVM

# $$y({\mathrm x})=\sum^N_{n=1}\alpha_n t_n k({\mathrm x}{\mathrm x_n})={\mathrm w}^T{\mathrm x}+b \\
# {\mathrm w}=\sum^N_{n=1}\alpha_n t_n \phi({\mathrm x}_n)$$
# 
# linear kernel:  
# $$k({\mathrm x}_i, {\mathrm x}_j) = \phi({\mathrm x}_i)^T\phi({\mathrm x}_j) = {\mathrm x}_i^T{\mathrm x}_j$$
# 
# polynomial kernel:  
# $$k({\mathrm x}_i, {\mathrm x}_j) = \phi({\mathrm x}_i)^T\phi({\mathrm x}_j) = ({\mathrm x}_i^T{\mathrm x}_j)^2 \\
# \phi({\mathrm x}) = [x^2_1, \sqrt{2}x_1x_2, x_2^2] \\
# {\mathrm x} = [x_1,x_2]$$

# In[16]:


def get_kernel(x, n):
    if n == 1:
        phi = np.asarray(x.T).T
    elif n == 2:
        phi = np.array([x.T[0]**2, np.sqrt(2)*x.T[0]*x.T[1], x.T[1]**2]).T
    return phi


# optimize $w,\ b,\ \{\xi\}$
# 
# $$\begin{aligned}
# \frac{\partial L}{\partial W} = 0 &\Rightarrow w = \sum^N_{n = 1}a_nt_n\phi({\mathrm x}_n) \\
# \frac{\partial L}{\partial b} = 0 &\Rightarrow \sum^N_{n=1}a_nt_n = 0 \\
# \frac{\partial L}{\partial \xi_n} = 0 &\Rightarrow a_n = C-\mu_n
# \end{aligned}$$

# In[17]:


def get_wb(a, t, x, n):
    """
    n: kernel param
    """
    if n == 1:
        phi_x = x
    elif n == 2:
        phi_x = get_kernel(x, n)
    at = a * t
    w = at.dot(phi_x)
    idx_s = np.where(a != 0)[0]
    idx_m = np.where(np.logical_and(0 < a, a < 1))[0]
    
    if len(idx_m) == 0:
        b = -1
    else:
        b = np.sum(t[idx_m]) - np.sum(np.linalg.multi_dot([at[idx_s], phi_x[idx_s], phi_x[idx_m].T]))
        b /= len(idx_m)
    return w, b


# In[18]:


def predict(W, B, x, label):
    phi_x = x
    pred = np.empty(len(phi_x))
    for i in range(len(phi_x)):
        cand = []
        for w, b, lab in zip(W, B, label):
            y = w.dot(phi_x[i]) + b
            if y > 0:
                cand.append(lab[0])
            else:
                cand.append(lab[1])
        pred[i] = Counter(cand).most_common(1)[0][0]
    return pred


# In[27]:


def get_mesh(x, y):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    return xx, yy


# linear

# In[28]:


clf = SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(train_x, train_t)

coef = np.abs(clf.dual_coef_)
sv_idx = clf.support_

alphas = np.zeros([len(train_x), 2])
alphas[sv_idx] = coef.T

idx_0 = np.where(train_t == 0)
idx_1 = np.where(train_t == 1)
idx_2 = np.where(train_t == 2)

label = ((0, 1), (0, 2), (1, 2))
t01 = np.array([ 1] * 100 + [-1] * 100 + [ 0] * 100)
t02 = np.array([ 1] * 100 + [ 0] * 100 + [-1] * 100)
t12 = np.array([ 0] * 100 + [ 1] * 100 + [-1] * 100)

a01 = np.concatenate((alphas[:200, 0], np.zeros(100)))
a02 = np.concatenate((alphas[:100, 1], np.zeros(100), alphas[200:, 0]))
a12 = np.concatenate((np.zeros(100), alphas[100:, 1]))

w01, b01 = get_wb(a01, t01, train_x, 1)
w02, b02 = get_wb(a02, t02, train_x, 1)
w12, b12 = get_wb(a12, t12, train_x, 1)

W = (w01, w02, w12)
B = (b01, b02, b12)

X0, X1 = train_x[:, 0], train_x[:, 1]
xx, yy = get_mesh(X0, X1)

Z = predict(W, B, np.column_stack((xx.flatten(), yy.flatten())), label)

Z = Z.reshape(xx.shape)
# print(Z)
fig, sub = plt.subplots(ncols = 2, figsize=(12, 8))
sub[0].contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.3)
sub[0].scatter(X0[sv_idx], X1[sv_idx], c='black', s=60, label='support vector')
sub[0].scatter(X0[idx_0], X1[idx_0], c='blue', s=50, marker='.', label='0')
sub[0].scatter(X0[idx_1], X1[idx_1], c='violet', s=50, marker='+', label='1')
sub[0].scatter(X0[idx_2], X1[idx_2], c='red', s=40, marker='d', label='2')
sub[0].legend()
sub[0].set_xlim(xx.min(), xx.max())
sub[0].set_ylim(yy.min(), yy.max())
sub[0].set_title('linear kernel')


clf2 = SVC(kernel='poly', degree = 2, decision_function_shape='ovo')
clf2.fit(train_x, train_t)

coef = np.abs(clf2.dual_coef_)
sv_idx = clf2.support_

alphas = np.zeros([len(train_x), 2])
alphas[sv_idx] = coef.T

idx_0 = np.where(train_t == 0)
idx_1 = np.where(train_t == 1)
idx_2 = np.where(train_t == 2)

label = ((0, 1), (0, 2), (1, 2))
t01 = np.array([ 1] * 100 + [-1] * 100 + [ 0] * 100)
t02 = np.array([ 1] * 100 + [ 0] * 100 + [-1] * 100)
t12 = np.array([ 0] * 100 + [ 1] * 100 + [-1] * 100)

a01 = np.concatenate((alphas[:200, 0], np.zeros(100)))
a02 = np.concatenate((alphas[:100, 1], np.zeros(100), alphas[200:, 0]))
a12 = np.concatenate((np.zeros(100), alphas[100:, 1]))

w01, b01 = get_wb(a01, t01, train_x, 2)
w02, b02 = get_wb(a02, t02, train_x, 2)
w12, b12 = get_wb(a12, t12, train_x, 2)

W = (w01, w02, w12)
B = (b01, b02, b12)

X0, X1 = train_x[:, 0], train_x[:, 1]
xx, yy = get_mesh(X0, X1)

Z = predict(W, B, np.vstack((xx.ravel()**2, np.sqrt(2)*xx.ravel()*yy.ravel(), yy.ravel()**2)).T, label)

Z = Z.reshape(xx.shape)


sub[1].contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.3)
sub[1].scatter(X0[sv_idx], X1[sv_idx], c='black', s=60, label='support vector')
sub[1].scatter(X0[idx_0], X1[idx_0], c='blue', s=50, marker='.', label='0')
sub[1].scatter(X0[idx_1], X1[idx_1], c='violet', s=50, marker='+', label='1')
sub[1].scatter(X0[idx_2], X1[idx_2], c='red', s=40, marker='d', label='2')
sub[1].legend()
sub[1].set_xlim(xx.min(), xx.max())
sub[1].set_ylim(yy.min(), yy.max())
sub[1].set_title('polynomial kernel')
plt.show();


# In[ ]:




