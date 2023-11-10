import pandas as pd
import pylab as plt
import warnings
warnings.filterwarnings('ignore')
plt.rc('font', family = 'SimHei', size = 10)
plt.rc('axes', unicode_minus = False)
data = pd.read_excel('处理完全的数据.xlsx')

#男女0-1
data['性别'][data['性别'] == '男'] = 1
data['性别'][data['性别'] == '女'] = 0

#删除性别？？
data = data.drop(labels=572, axis = 0)

import numpy as np
from sklearn.mixture import GaussianMixture
# 构造GMM模型
gmm = GaussianMixture(n_components=4)

# 拟合数据集
gmm.fit(data)

# 计算每个数据点属于每个高斯分布的概率值
scores = gmm.score_samples(data)

# 设置阈值，判断异常值
threshold = np.percentile(scores, 5)
data_ = data[scores > threshold]

data_.to_excel('高斯混合模型.xlsx')