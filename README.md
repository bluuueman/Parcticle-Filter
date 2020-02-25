# Parcticle-Filter
a localization algorithm based on particle filter<br>

* 关于particle filtering（粒子滤波），对于定位问题，本质上是蒙特卡洛法。步骤如下：<br>
  1.随机生成粒子（create）<br>
  2.预测状态（predic）：robot移动后同步更新粒子状态<br>
  3.更新粒子权重（update）：根据粒子和robot的相对距离(参照地标)，更新权重<br>
  4.重采样（resample）：增加高权值粒子数量，减少低权值粒子数<br>
  5.估计robot位置（estimate）：通过粒子位置的加权平均数可计算出robot的大致位置。<br>
  6.在robot运动过程中重复2到5步骤<br>

### 问题1：基于particle位置，计算最终定位robot的唯一位置并输出到屏幕

* estimate函数采用加权平均数计算particle位置得出robot最终位置
```python
#line127-line131
def estimate(particles, weights):
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean
#line177
estimate_position=estimate(particles,weights)
```
* 输出到屏幕
```python
#line178
cv2.circle(img, ((int(estimate_position[0])),(int(estimate_position[1]))), 4,(0,0,255), -1)
#line182
cv2.circle(img, (10, 75), 4, (0, 0, 255), -1)
#line186
cv2.putText(img, "Robot Estimated Position", (30, 80), 1, 1.0, (0, 0, 255))
```

### 问题2：采用帕累托分布代替正态分布修改权重

* 修改update函数中particle的权重计算方式
```python
#line96
weights *= scipy.stats.pareto(1).pdf(0.1*abs(z[i]-distance)+1，1)
```
由于帕累托分布中x大于等于1，令
`x=abs(z[i]-distance)+1`
同时为了保证粒子多样性，使结果更加准确，在绝对值前乘以系数0.1即
`x=0.1*abs(z[i]-distance)+1`

### 问题3：landmark和robot的距离增加随机误差

* 原本代码中已包含随机误差
```python
#line52
zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
```
* 原代码中关于重采样的粒子也已包含噪声
```python
#line81
dist = (u[1] * dt) + (np.random.randn(N) * std[1])
```

* 由于采用加权平均值方式对robot位置进行估计，在粒子数足够多，环境相对简单的情况下，实际噪声和随机误差对结果影响并不明显

### 其他

* 由于环境使用的opencv版本为4.x，对部分版本相关代码有所修改
```python
#line19
LINE_AA = cv2.LINE_AA if cv2.__version__[0] == '4' else cv2.CV_AA
```
### 可能的优化
* 控制重采样频率，权重偏向小数量样本时，应该进行重采样；在粒子权重偏向大数量样本时，降低采样频率，以减小状态多样性的丢失。
