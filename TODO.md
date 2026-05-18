# 说明

# 模型说明

## PINNs 结构

Physics-Informed Neural Networks, PINNs (物理信息神经网络)

### 网络结构

建立一个全连接神经网络

输入层: 三个神经元 时间 t 摆长 l 初始角度 $\theta_0$

隐藏层: 3 层，每层 128 个神经元，Tanh 激活函数

输出层: 一个神经元 摆角 $\theta$

## 损失函数

PINNs 的损失函数表示为:
$$
Loss(\theta) = \underbrace{\lambda_{data} \frac{1}{N_u} \sum_{i=1}^{N_u} |\hat{u}_i - u_i|^2}_{Loss_{data}} + \underbrace{\lambda_{f} \frac{1}{N_f} \sum_{j=1}^{N_f} |f(x_j, t_j)|^2}_{Loss_{physics}}
$$

# 模型1 

## 方程基础

$$\frac{d^2\theta}{dt^2} + \frac{g}{l} \sin\theta = 0$$

## 损失函数

$$
Loss_{physics} = \frac{1}{N_f} \sum_{j=1}^{N_f} \left| \frac{\partial^2 \theta_j}{\partial t^2} + \frac{g_j}{l_j} \sin(\theta_j) \right|^2
$$

# 模型2

## 方程基础

小角度近似 ($\theta_0 < 5^\circ$)

$$
\theta(t) = \theta_0 \cos\left(\sqrt{\frac{g}{l}}\, t\right)
$$

## 损失函数

$$
Loss_{physics} = \frac{1}{N_f} \sum_{j=1}^{N_f} \left| \frac{\partial^2 \theta_j}{\partial t^2} + \frac{g_j}{l_j} \theta_j \right|^2
$$


# 模型实现
intuition_physics_ml\models\pinns\pinns_loss.py PINNs 的损失函数

intuition_physics_ml\models\pinns\pinns_net.py PINNs 的网络结构

# 模型训练

intuition_physics_ml\training\pinns\pinns_dataloader.py PINNs 的数据加载器

intuition_physics_ml\training\pinns\pinns_config.py PINNs 的超参数配置

intuition_physics_ml\training\pinns\pinns_trainer.py PINNs 的训练器

# 模型存储

artifacts

# 数据集

data\raw\algorithm_prediction_data\ode_prediction_data.csv 训练集

无验证集

data\test_dataset\ode_prediction_test.csv 测试集

# 模型评估

notebooks\PINNs_visualization.ipynb 可视化 + 评估 

