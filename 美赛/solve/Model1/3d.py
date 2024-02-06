import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np
from scipy.integrate import odeint

def S(y, t):
    M, F, P, A = y
    c = 0.08 #生殖率
    k = 0.01 #控制性转率
    N = M + F # 总数
    b = M / N #雄占比
    B = b * (1 - b) * c # 繁殖率
    Z = 0.30 / (2 + y[2]**k) + 0.53 #转雄概率
    rp, da = 0.09, 0.004# P生存率, a死亡率
    dm, df = 0.045, 0.035 #死亡率
    bm, bf, ba = 0.5, 0.4, 0.05 #捕获能力
    ap, aa = 0.002, 0.002#被捕率
    dM = B * Z * N - dm * M + bm * ap * P * M - aa * ba * A * M + 0.01
    dF = B * (1 - Z) * N - df * F + bf * ap * P * F - aa * ba * A * F - 0.085
    dP = rp * P - ap * P * N
    dA = - da * A + ba * aa * A * N
    return [dM, dF, dP, dA]

def S1(y, t):
    M, F, P, A = y
    c = 0.08 #生殖率
    k = 0.01 #控制性转率
    N = M + F # 总数
    b = M / N #雄占比
    B = b * (1 - b) * c # 繁殖率
    Z = 0.5#0.30 / (2 + y[2]**k) + 0.53 #转雄概率
    rp, da = 0.09, 0.004# P生存率, a死亡率
    dm, df = 0.045, 0.045 #死亡率
    bm, bf, ba = 0.5, 0.5, 0.05 #捕获能力
    ap, aa = 0.002, 0.002#被捕率
    dM = B * Z * N - dm * M + bm * ap * P * M - aa * ba * A * M + 0.01
    dF = B * (1 - Z) * N - df * F + bf * ap * P * F - aa * ba * A * F - 0.085
    dP = rp * P - ap * P * N
    dA = - da * A + ba * aa * A * N
    return [dM, dF, dP, dA]

# 初始条件
y0 = [40, 30, 50, 20]

# 时间点
t = np.linspace(0, 1000, 100)

# 解ODE系统
M, F, P, A = odeint(S, y0, t).T

M1, F1, P1, A1 = odeint(S1, y0, t).T
# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D线性图

ax.plot(F1 + M1, P1, A1)
ax.plot(F + M, P, A, color = 'red')

# 设置坐标轴标签
ax.set_xlabel('Lamprey')
ax.set_ylabel('Prey')
ax.set_zlabel('Human')
ax.legend(['No sex Change', 'sex Change'])
#ax.set_title('Phase trajectories of lamprey-human-prey population changes')
# 显示图形
plt.show()