import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_importance(W):
    importance = np.linalg.norm(W, axis=0)
    return importance

def calculate_mask(importance, percentile=80):
    threshold = np.percentile(importance, percentile)
    mask = importance < threshold
    return mask

def main():
    for i in range(32):
        weight_tensor_name = 'blk_' + str(i) + '_' + 'ffn_down'
        activation_tensor_name = f'ffn_gate_par-{i}'
        df = pd.read_csv('./weights/' + weight_tensor_name + '_weight', sep='\s+', header=None)

        W = df.values

        if W.shape != (4096, 14336):
            raise ValueError('Tensor shape incorrect!!')

        importance = calculate_importance(W)
        print('Importance vector:\n', importance)

        importance.tofile('./importance/' + activation_tensor_name + '_importance')

        mask = calculate_mask(importance)
        mask.astype(np.int8).tofile('./masks/' + activation_tensor_name + '_masks')  # 保存mask数组

        sorted_importance = np.sort(importance)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_importance, marker='o')
        plt.title('Importance Values Distribution')
        plt.xlabel('Index (sorted)')
        plt.ylabel('Importance')
        plt.grid(True)
        plt.savefig('./visualization/' + activation_tensor_name + '_importance.png')

if __name__ == '__main__':
    main()
