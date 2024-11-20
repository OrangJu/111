---
marp: true
theme: default
class: lead
backgroundImage: url('background.png') 
paginate: true
header: "损失函数"
footer: "基于PyTorch的实现"
---
---
# 损失函数

理解和实现常见的损失函数  
研究深度学习中的关键模块
---
---
# 均方误差 (Mean Squared Error, MSE)

## 基本理论
均方误差用于衡量预测值和真实值之间的平方差平均值。  
公式为：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

- **优点**：对预测误差较大的情况给予更高惩罚  
- **缺点**：对异常值较敏感

---

## PyTorch 实现

```python
# 之后的代码省略import
import torch
import torch.nn as nn

# 创建真实值和预测值
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.1, 2.0, 2.9])

# 使用nn.MSELoss
mse_loss = nn.MSELoss()
loss = mse_loss(y_pred, y_true)
print("MSE Loss:", loss.item())
```