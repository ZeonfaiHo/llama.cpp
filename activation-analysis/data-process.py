# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 替换成您的文件路径
# file_path = './silu'

# # 尝试读取空格分隔的文件
# data = pd.read_csv(file_path, delim_whitespace=True)

# # 计算每一列的方差
# variances = data.var()
# variances = variances.sort_values()

# # 计算方差的对数，排除方差为0的情况
# log_variances = np.log(variances[variances > 0])

# # 创建横坐标（百分比）
# x_labels = np.linspace(0, 100, len(log_variances))

# # 绘制图表
# plt.figure(figsize=(12, 6))
# plt.plot(x_labels, log_variances)
# plt.xlabel('Percentage (%)')
# plt.ylabel('Log Variance')
# plt.title('Log Variance of Each Column')
# plt.grid(True)
# # plt.show()
# plt.savefig('visualize.png')


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 替换成您的文件路径
# file_path = './silu_x_result_w3'

# # 读取数据
# data = pd.read_csv(file_path, delim_whitespace=True)

# # 存储每列最大化元素数量的比例
# max_proportions = []
# col = 0

# for column in data.columns:
#     if col % 10 == 0:
#         print(col)
    
#     col += 1

#     max_count = 0
#     for potential_center in data[column]:
#         # 计算在潜在中心值±0.1范围内的元素数量
#         count = ((data[column] >= potential_center - 0.1) & 
#                  (data[column] <= potential_center + 0.1)).sum()
#         # 更新最大数量
#         if count > max_count:
#             max_count = count

#     # 计算比例
#     proportion = max_count / len(data[column])
#     max_proportions.append(proportion)

#     if col == 1000:
#         break

# # 创建横坐标（百分比）
# max_proportions.sort()
# x_labels = np.linspace(0, 100, len(max_proportions))

# # 绘制图表
# plt.figure(figsize=(12, 6))
# plt.plot(x_labels, max_proportions)
# plt.xlabel('Sorted Columns')
# plt.ylabel('Maximum Proportion Within ±0.1')
# plt.title('Maximum Proportion of Elements Within ±0.1 of Center Value for Each Column')
# plt.grid(True)
# plt.savefig(file_path + '.png')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_max_proportions(file_path, tolerance):
    """
    计算并绘制每列数据中，最大比例的元素在指定扰动范围内的分布。

    :param file_path: 数据文件的路径。
    :param tolerance: 允许的扰动范围。
    """
    # 读取数据
    data = pd.read_csv(file_path, delim_whitespace=True)

    # 存储每列最大化元素数量的比例
    max_proportions = []
    col = 0

    for column in data.columns:
        if col % 10 == 0:
            print(col)
        
        col += 1

        max_count = 0
        for potential_center in data[column]:
            # 计算在潜在中心值±tolerance范围内的元素数量
            count = ((data[column] >= potential_center - tolerance) & 
                     (data[column] <= potential_center + tolerance)).sum()
            # 更新最大数量
            if count > max_count:
                max_count = count

        # 计算比例
        proportion = max_count / len(data[column])
        max_proportions.append(proportion)

        if col == 1000:
            break

    # 创建横坐标（百分比）
    max_proportions.sort()
    x_labels = np.linspace(0, 100, len(max_proportions))

    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(x_labels, max_proportions)
    plt.xlabel('Sorted Columns')
    plt.ylabel(f'Maximum Proportion Within ±{tolerance}')
    plt.title(f'Maximum Proportion of Elements Within ±{tolerance} of Center Value for Each Column')
    plt.grid(True)
    plt.savefig(file_path + '_tolerance_' + str(tolerance) + '.png')

# 替换成您的文件路径
file_path = './inpFF'
# 设置扰动范围
tolerance = 0.0001  # 可以根据需要调整这个值

# 调用函数
calculate_max_proportions(file_path, tolerance)
