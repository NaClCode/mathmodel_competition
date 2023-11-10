# %%
import numpy as np
import pandas as pd
from scipy.stats import kstest
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.special import inv_boxcox
def KsNormDetect(df:pd.DataFrame):
    # 计算均值
    u = df['qty'].mean()
    # 计算标准差
    std = df['qty'].std()
    # 计算P值
    print(kstest(df['qty'], 'norm', (u, std)))
    res = kstest(df['qty'], 'norm', (u, std))[1]
    print('均值为：%.2f, 标准差为：%.2f' % (u, std))
    # 判断p值是否服从正态分布，p<=0.05 拒绝原假设 不服从正态分布
    if res <= 0.05:
        print('该列数据不服从正态分布')
        return True
    else:
        print('该列数据服从正态分布')
        return False
def OutlierDetection(df:pd.DataFrame):
    # 计算均值
    u = df.mean()
    # 计算标准差
    std = df.std()

    # 定义3σ法则识别异常值
    outliers = df[np.abs(df - u) > 3 * std]
    # 剔除异常值，保留正常的数据
    clean_data = df[np.abs(df - u) < 3 * std]
    # 返回异常值和剔除异常值后的数据
    return outliers, clean_data



# %%
import warnings
warnings.filterwarnings("ignore")
for i in range(1996):

    print("-" * 66)
    print(f'时间序列{i}')
    # 可以转换为pandas的DataFrame 便于调用方法计算均值和标准差
    df = pd.read_csv(f'time/time{i}.csv')
    # K-S检验
    ks_res = KsNormDetect(df)
    
    outliers, clean_data = OutlierDetection(df['qty'])
    # 异常值和剔除异常值后的数据
    df.loc[outliers.index.to_list(),'qty'] = clean_data.mean()
    if outliers.shape[0] != 0:
        print('异常值:')
        print(outliers)
    else: print('无异常值')
    df.to_csv(f'No_abnormality_time/time{i}.csv', index=False)


