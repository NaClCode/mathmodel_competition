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

# 初始条件
y0 = [40, 30, 50, 20]

# 时间点
t = np.linspace(0, 1000, 100)

# 解ODE系统
M, F, P, A = odeint(S, y0, t).T

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(t, M, label='Male Sea Lampreys', color='blue')
plt.plot(t, F, label='Female Sea Lampreys', color='red')
plt.plot(t, P, label='Prey', color='grey')
plt.plot(t, A, label='People', color='green')
#plt.plot(time, M + F, label='Total Population', color='black')
plt.xlabel('Time')
plt.ylabel('Population')
#plt.title('Structured Population Model Simulation')
plt.legend()
plt.savefig(fname = 'population_model_simulation.jpg', dpi = 500, bbox_inches = 'tight')
 
   
   
   
   