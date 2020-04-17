import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle
import os

os.chdir("C:/Users/user/Documents/py/00_NCTU_ML")

data_x = pd.read_csv("dataset_X.csv").iloc[:, 1:]
data_t = pd.read_csv("dataset_T.csv").iloc[:, 1:]


def zero(x):
    intercept = np.repeat(1, len(x), axis=0).transpose()
    return pd.DataFrame({"C": intercept}).astype("float32")


def one(x):
    return x


def two(x):
    sq = pd.DataFrame()
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if j >= i:
                row = []
                for k in range(x.shape[0]):
                    row.extend([x.iloc[k, i] * x.iloc[k, j]])
                str = data_x.columns.values[i] + "*" + data_x.columns.values[j]
                sq[str] = row
    return sq


def solve_reg(x, y, M):
    y = y.values

    inter = zero(x).reset_index(drop=True)
    onep = one(x).reset_index(drop=True)
    sq = two(x).reset_index(drop=True)

    if M == 0:
        phi = pd.concat([inter], axis=1, ignore_index=True).values
    elif M == 1:
        phi = pd.concat([inter, onep], axis=1, ignore_index=True).values
    elif M == 2:
        phi = pd.concat([inter, onep, sq], axis=1, ignore_index=True).values
    else:
        return print("invalid M")

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(phi.T, phi)), phi.T), y)
    return (w, phi)


def get_ID(dataset):
    list0 = [x for x in range(dataset.shape[0])]
    shuffle(list0)
    return list0


def split_ID(list0, val_size: float):
    return [
        list0[i : i + int(val_size * len(list0))]
        for i in range(0, len(list0), int(val_size * len(list0)))
    ]


def split_ID2(list0, piles):
    p = [[] for _ in range(piles)]
    for i in range(len(list0)):
        for j in range(piles):
            if i % piles == j:
                p[j].append(list0[i])
    return [p][0]


def cross_val(x, t, piles, nd):
    p = split_ID2(get_ID(x), piles)
    RMSE = []
    for i in range(len(p)):
        cv_x = x.loc[p[i]]
        cv_t = t.loc[p[i]]
        sub_x = x.loc[~data_x.index.isin(p[i])]
        sub_t = t.loc[~data_x.index.isin(p[i])]
        w, phi = w, phi = solve_reg(sub_x, sub_t, nd)
        res = (
            np.ones(sub_t.shape[0]).dot((np.matmul(phi, w) - sub_t.values) ** 2)
            / sub_t.shape[0]
        ) ** 0.5
        RMSE.append(res[0])

    return RMSE


p = [cross_val(data_x, data_t, 5, 1), cross_val(data_x, data_t, 5, 2)]
print(p)
# ! RMSE
