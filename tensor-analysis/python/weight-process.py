import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 计算向量 X 的每个位置的重要性
def calculate_importance(W):
    importance = np.linalg.norm(W, axis=0)
    return importance

def main():
    for i in range(32):
        tensor_name = 'ffn_down'
        df = pd.read_csv('./weights/blk_' + str(i) + '_' + tensor_name + '_weight', sep='\s+', header=None)

        # 将 DataFrame 转换为 NumPy 数组
        W = df.values

        # if W.shape != (14336, 4096):
        if W.shape != (4096, 14336):
            raise ValueError('Tensor shape incorrect!!')

        importance = calculate_importance(W)

        # 打印重要性向量
        print('Importance vector:\n', importance)

        # 将重要性向量保存为二进制浮点数数组
        importance.tofile('./importance/' + 'blk_' + str(i) + '_' + tensor_name + '_importance')

        # 排序重要性数值并绘图
        sorted_importance = np.sort(importance)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_importance, marker='o')
        plt.title('Importance Values Distribution')
        plt.xlabel('Index (sorted)')
        plt.ylabel('Importance')
        plt.grid(True)
        plt.savefig('./visualization/importance_distribution.png')  # 保存图像

if __name__ == '__main__':
    main()
