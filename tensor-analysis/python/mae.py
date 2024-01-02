# import pandas as pd
# import numpy as np

# # 读取文件
# ffn_gate_par_16 = pd.read_csv('./raw-activations/ffn_gate_par-16', delim_whitespace=True, header=None).values
# ffn_gate_par_16_masks = pd.read_csv('./raw-activations/ffn_gate_par-16_masks', delim_whitespace=True, header=None).values
# ffn_gate_par_16_no_masks = pd.read_csv('./raw-activations/ffn_gate_par-16_no_masks', delim_whitespace=True, header=None).values

# # 计算MAE
# def calculate_mae(matrix1, matrix2):
#     return np.mean(np.abs(matrix1 - matrix2))

# mae_masks = calculate_mae(ffn_gate_par_16, ffn_gate_par_16_masks)
# mae_no_masks = calculate_mae(ffn_gate_par_16, ffn_gate_par_16_no_masks)

# print(f"MAE between ffn_gate_par-16 and ffn_gate_par-16_masks: {mae_masks}")
# print(f"MAE between ffn_gate_par-16 and ffn_gate_par-16_no_masks: {mae_no_masks}")

# # 计算相同元素的比例
# def calculate_similarity(matrix1, matrix2):
#     return np.mean(matrix1 == matrix2)

# similarity_masks = calculate_similarity(ffn_gate_par_16, ffn_gate_par_16_masks)
# similarity_no_masks = calculate_similarity(ffn_gate_par_16, ffn_gate_par_16_no_masks)

# print(f"Similarity between ffn_gate_par-16 and ffn_gate_par-16_masks: {similarity_masks}")
# print(f"Similarity between ffn_gate_par-16 and ffn_gate_par-16_no_masks: {similarity_no_masks}")

# # 分析不同元素
# def analyze_differences(matrix1, matrix2):
#     differences = matrix1 - matrix2
#     return differences[np.nonzero(differences)]

# differences_masks = analyze_differences(ffn_gate_par_16, ffn_gate_par_16_masks)
# differences_no_masks = analyze_differences(ffn_gate_par_16, ffn_gate_par_16_no_masks)

# print(f"Number of different elements in masks: {len(differences_masks)}")
# print(f"Number of different elements in no_masks: {len(differences_no_masks)}")

import pandas as pd
import numpy as np

# 读取文件
ffn_out_16 = pd.read_csv('./raw-activations/result_output', delim_whitespace=True, header=None).values
ffn_out_16_masks = pd.read_csv('./raw-activations/result_output_masks', delim_whitespace=True, header=None).values
ffn_out_16_no_masks = pd.read_csv('./raw-activations/result_output_no_masks', delim_whitespace=True, header=None).values

# 计算MAE
def calculate_mae(matrix1, matrix2):
    return np.mean(np.abs(matrix1 - matrix2))

mae_masks = calculate_mae(ffn_out_16, ffn_out_16_masks)
mae_no_masks = calculate_mae(ffn_out_16, ffn_out_16_no_masks)

print(f"MAE between result_output and result_output_masks: {mae_masks}")
print(f"MAE between result_output and result_output_no_masks: {mae_no_masks}")
