# import pandas as pd

# # Load the data from the uploaded file
# file_path = 'inpFF'
# data = pd.read_csv(file_path, delimiter=' ', header=None)

# # Flatten the data into a single series for analysis
# flattened_data = data.values.flatten()

# # Display the first few values for an initial inspection
# flattened_data[:10]

# import matplotlib.pyplot as plt
# import numpy as np

# # Calculate the cumulative distribution function (CDF)
# sorted_data = np.sort(flattened_data)
# cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

# # Plot the CDF
# plt.figure(figsize=(10, 6))
# plt.plot(sorted_data, cdf)
# plt.xlabel('value')
# plt.ylabel('partition')
# plt.title('distribution')
# plt.grid(True)

# plt.savefig(file_path + '_element.png')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据文件的路径
file_path = 'inpFF'  # 替换为你的数据文件路径

# 从文件中加载数据，假设它是空格分隔的
data = pd.read_csv(file_path, delimiter=' ', header=None)

# 检查数据形状以确保正确加载
print("Data shape:", data.shape)

# 将数据展平为单个序列进行分析
flattened_data = data.values.flatten()
print(len(flattened_data))

# 计算累积分布函数（CDF）
sorted_data = np.sort(flattened_data)
cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

# 绘制CDF
plt.figure(figsize=(10, 6))
plt.plot(cdf, sorted_data)
plt.yscale('log')
plt.xlabel('accumulate proportion')
plt.ylabel('value')
plt.title('element value distribution')
plt.grid(True)
plt.savefig(file_path + '_elements_distribution.png')
