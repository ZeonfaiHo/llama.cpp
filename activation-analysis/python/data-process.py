# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import struct

# def calculate_max_proportions(file_name, tolerance):
#     """
#     计算并绘制每列数据中，最大比例的元素在指定扰动范围内的分布。

#     :param file_path: 数据文件的路径。
#     :param tolerance: 允许的扰动范围。
#     """
#     # 读取数据
#     file_path = './raw-activations/' + file_name
#     data = pd.read_csv(file_path, delim_whitespace=True)

#     # 存储每列最大化元素数量的比例
#     max_proportions = []
#     col = 0

#     centers = {}

#     for column in data.columns:
#         if col % 10 == 0:
#             print(col)
        
#         col += 1

#         max_count = 0
#         for potential_center in data[column]:
#             # 计算在潜在中心值±tolerance范围内的元素数量
#             count = ((data[column] >= potential_center - tolerance) & 
#                      (data[column] <= potential_center + tolerance)).sum()
#             # 更新最大数量
#             if count > max_count:
#                 max_count = count
#                 centers[column] = potential_center

#         # 计算比例
#         proportion = max_count / len(data[column])
#         max_proportions.append(proportion)

#         if col == 100:
#             break

#     # 创建横坐标（百分比）
#     max_proportions.sort()
#     x_labels = np.linspace(0, 100, len(max_proportions))

#     # 绘制图表
#     plt.figure(figsize=(12, 6))
#     plt.plot(x_labels, max_proportions)
#     plt.xlabel('Sorted Columns')
#     plt.ylabel(f'Maximum Proportion Within ±{tolerance}')
#     plt.title(f'Maximum Proportion of Elements Within ±{tolerance} of Center Value for Each Column')
#     plt.grid(True)
#     plt.savefig('./visualization/' + file_name + '_tolerance_' + str(tolerance) + '.png')

#     with open('./centers/' + file_name + '_centers', 'wb') as f:
#         for center in centers.values():
#             # 使用'f'格式化符号来表示单精度浮点数
#             f.write(struct.pack('f', center))

# for i in range(32):
#     file_name = 'ffn_norm_layer_' + str(i)
#     # 设置允许的扰动范围
#     tolerance = 0.3
#     calculate_max_proportions(file_name, tolerance)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct

def calculate_median_centers(data):
    centers = {}
    for column in data.columns:
        centers[column] = data[column].median()
    return centers

def calculate_max_proportions_with_centers(data, centers, tolerance):
    max_proportions = []
    for column in data.columns:
        center = centers[column]
        count = ((data[column] >= center - tolerance) &
                 (data[column] <= center + tolerance)).sum()
        proportion = count / len(data[column])
        max_proportions.append(proportion)
    return max_proportions

def find_optimal_tolerance(data, target_proportion=0.8, lower_bound=0.0, upper_bound=10):
    centers = calculate_median_centers(data)
    while upper_bound - lower_bound > 0.01:
        mid_tolerance = (lower_bound + upper_bound) / 2

        print('testing tolerance ' + str(mid_tolerance))

        max_proportions = calculate_max_proportions_with_centers(data, centers, mid_tolerance)
        columns_meeting_condition = np.sum(np.array(max_proportions) >= target_proportion)
        proportion_of_columns = columns_meeting_condition / len(max_proportions)

        print('result: ' + str(proportion_of_columns))    

        if 0.75 <= proportion_of_columns <= 0.85:
            break
        elif proportion_of_columns < 0.75:
            lower_bound = mid_tolerance
        else:
            upper_bound = mid_tolerance

    ret = (lower_bound + upper_bound) / 2
    max_proportions = calculate_max_proportions_with_centers(data, centers, mid_tolerance)

    for i in range(len(max_proportions)):
        if max_proportions[i] < 0.8:
            centers[i] = -1e9

    return ret

def visualize_and_save(data, tolerance, file_name):
    centers = calculate_median_centers(data)
    max_proportions = calculate_max_proportions_with_centers(data, centers, tolerance)
    max_proportions = calculate_max_proportions_with_centers(data, centers, tolerance)
    x_labels = np.linspace(0, 100, len(max_proportions))

    plt.figure(figsize=(12, 6))
    plt.plot(x_labels, sorted(max_proportions))
    plt.xlabel('Sorted Columns')
    plt.ylabel(f'Maximum Proportion Within ±{tolerance}')
    plt.title(f'Maximum Proportion of Elements Within ±{tolerance} of Center Value for Each Column')
    plt.grid(True)
    plt.savefig('./visualization/' + file_name + '_tolerance_' + str(tolerance) + '.png')

    # with open('./centers/' + file_name + '_tolerance_' + str(tolerance) + '_centers', 'wb') as f:

    for i in range(len(max_proportions)):
        if max_proportions[i] < 0.8:
            centers[i] = -1e9

    with open('./centers/' + file_name + '_centers', 'wb') as f:
        for center in centers.values():
            f.write(struct.pack('f', center))

activation_name = 'ffn_norm'

with open(activation_name + '_tolerances', 'wb') as tolerance_file:
    for i in range(32):
        file_name = activation_name + '_layer_' + str(i)
        print('processing ' + file_name)
        
        file_path = './raw-activations/' + file_name
        data = pd.read_csv(file_path, delim_whitespace=True)
        optimal_tolerance = find_optimal_tolerance(data)

        print('optimal tolerance: ' + str(optimal_tolerance))

        # 将最优扰动范围写入二进制文件
        tolerance_file.write(struct.pack('f', optimal_tolerance))

        visualize_and_save(data, optimal_tolerance, file_name)