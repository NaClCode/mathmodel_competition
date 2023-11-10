import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import xgboost
import numpy as np
import pylab as plt
plt.rc('font', family = 'SimHei', size = 8)
plt.rc('axes', unicode_minus = False)
data_ = pd.read_excel('高斯混合模型.xlsx', index_col=0)
#x, y
y = data_['血糖']
x = data_.drop(labels=['血糖'], axis = 1).values

#画出xgboost的特征重要性图
xgb_reg = Pipeline([
    ('std_scaler', StandardScaler()),
    ('xgb_reg', xgboost.XGBRegressor(max_depth = 4))
])
xgb_reg.fit(x, y)
pd.DataFrame(xgb_reg['xgb_reg'].feature_importances_, index=data_.columns.drop(labels = '血糖'), columns=['特征重要性']).sort_values(by = ['特征重要性']).plot.barh()
plt.tight_layout()
plt.savefig('xgboost的特征重要性图.png', dpi = 500)

#xgboost+特征删除
r2 = []
del_col = data_.columns.drop('血糖')
del_column = []
x_ = x
for i in range(0, 29):
    xgb_reg = Pipeline([
        ('std_scaler', StandardScaler()),
        ('xgb_reg', xgboost.XGBRegressor(max_depth = 4))
    ])
    xgb_reg.fit(x_, y)
    r2.append(r2_score(y, xgb_reg.predict(x_)))
    del_num = np.argmin(xgb_reg['xgb_reg'].feature_importances_)
    del_col = np.delete(del_col, del_num)
    del_column.append(del_col)
    x_ = np.delete(x_, del_num, axis = 1)

#绘制删除特征数量与R^2的图片
#这样可以确定要删多少特征
r2 = np.array(r2)
import pylab as plt
plt.figure(figsize=(8, 8))
pd.DataFrame(r2, columns=['r2_score']).plot.line()
plt.tight_layout()
plt.savefig('删除特征数量与R^2(可以确定要删多少特征).png', dpi = 500)

print(del_column[18])

#spearman系数
data_ = data_.astype('float64')

plt.figure(figsize=(10, 10))
corr = data_.corr(method='spearman')['血糖'].drop(labels = ['血糖'], axis = 0)
pd.DataFrame(corr.values, index=corr.index, columns=['spearman']).sort_values(by=['spearman']).plot.barh()
plt.tight_layout()
plt.savefig('Spearman.png', dpi = 500)