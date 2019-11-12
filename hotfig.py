import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
np.random.seed(0)
uniform_data = np.random.rand(10, 12)
# 改变颜色映射的值范围
ax = sns.heatmap(uniform_data, cmap='YlGnBu')
plt.savefig("ax.png")
