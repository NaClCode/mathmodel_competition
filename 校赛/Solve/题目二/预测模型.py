import pandas as pd
data = pd.read_excel('高斯混合模型.xlsx')[['*天门冬氨酸氨基转换酶', '尿酸', '年龄', '性别', '甘油三酯', '红细胞体积分布宽度', '红细胞平均体积',
       '红细胞计数', '血小板平均体积', '血红蛋白', '血糖']]


x = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.preprocessing import MinMaxScaler
x = MinMaxScaler().fit_transform(x)

#划分训练集与测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import xgboost
ret = []
for i in range(0, 10):
    tmp = []
    for model in [LinearRegression(), SVR(), KNeighborsRegressor(), xgboost.XGBRegressor(min_child_weight = 10, eta = 0.05, colsample_bytree = 0.5, n_estimators = 136), MLPRegressor()]:
        model.fit(x_train, y_train)
        tmp.append(mean_squared_error(y_test, model.predict(x_test)) ** 0.5)
    ret.append(tmp)

import numpy as np
import pylab as plt
pd.DataFrame(np.array(ret), index = range(0, 10), columns=['LinearRegress', 'SVM', 'KNN', 'xgboost', 'BP']).plot.barh()
plt.savefig('模型比较.png', dpi = 500)

import xgboost
from sko.PSO import PSO
def f(params):
    model = xgboost.XGBRegressor(min_child_weight = int(params[0]), eta = params[1], colsample_bytree = params[2], n_estimators = int(params[3]), alpha = params[4])
    model.fit(x_train, y_train)
    rmse_test = mean_squared_error(y_test, model.predict(x_test)) ** 0.5
    rmse_train = mean_squared_error(y_train, model.predict(x_train)) ** 0.5
    rmse = abs(rmse_test - rmse_train)
    print(rmse_test)
    return rmse_test
    
pso = PSO(func=f, dim = 5, lb = [5, 0, 0, 200, 0], ub = [20, 1, 1, 500, 1], pop=10, max_iter=100)
pso_ret = pso.run()

print('粒子群算法最优参数', pso_ret.gbest_x)
plt.figure(figsize=(10, 10))
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train), 50):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown
model = xgboost.XGBRegressor(min_child_weight = int(pso_ret.gbest_x[0]))
import joblib
joblib.dump(model.fit(x, y), "xgboost.pkl")                       
plot_learning_curves(xgboost.XGBRegressor(min_child_weight = int(pso_ret.gbest_x[0]), eta = pso_ret.gbest_x[1], colsample_bytree = pso_ret.gbest_x[2], n_estimators = int(int(pso_ret.gbest_x[3]))), x, y)
plt.tight_layout()
plt.savefig('学习曲线.png', dpi = 500)

plt.figure(figsize=(10, 10))
plt.plot(pso_ret.gbest_y_hist)
plt.tight_layout()
plt.savefig('PSO.png', dpi = 500)

print('粒子群算法最优参数', pso_ret.gbest_x)
