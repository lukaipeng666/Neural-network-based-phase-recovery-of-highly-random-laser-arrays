# import matplotlib.pyplot as plt
# import pickle

# with open('distances_GS.pkl', 'rb') as f:
#     loaded_distances = pickle.load(f)
#
#     plt.hist(loaded_distances, bins=60, edgecolor='black')  # bins参数可以根据需要调整
#     plt.title('NEU_out_reverse of Distances')
#     plt.xlabel('Distance')
#     plt.ylabel('Frequency')
#     plt.grid(True)  # 添加网格线以便于观察
#     plt.show()

#
# # 加载数据
# with open('ssim_value_NEU.pkl', 'rb') as file:
#     data = pickle.load(file)
#
# # 确保数据是一个列表或类似列表的结构
# if not isinstance(data, (list, tuple)):
#     raise ValueError("加载的数据不是列表或元组形式，无法绘制箱线图。")
#
# # 绘制箱线图
# plt.boxplot(data, vert=True)  # vert=False 表示水平箱线图，如果需要垂直的则去掉这个参数或设为 True
#
# # 添加标题和轴标签
# plt.title('NEU_recovery_ssim')
# plt.xlabel('x')
# plt.ylabel('y')  # 对于水平箱线图，ylabel 实际上并不表示具体的意义，可以根据需要调整或省略
#
# # 显示图形
# plt.show()

import pickle
import numpy as np
#
# with open('mse_NEU.pkl', 'rb') as file:
#     data = pickle.load(file)
#
# mean_mse_GS_value = np.mean(data)
# print('mean_mse_NEU_value：{}'.format(mean_mse_GS_value))
#
# with open('distances_NEU.pkl', 'rb') as file:
#     data = pickle.load(file)
#
# mean_mse_GS_value = np.mean(data)
# print('mean_distances_NEU_value：{}'.format(mean_mse_GS_value))
#
with open('ssim_value_GS.pkl', 'rb') as file:
    data = pickle.load(file)

mean_mse_GS_value = np.mean(data)
print('mean_ssim_value_GS_value：{}'.format(mean_mse_GS_value))