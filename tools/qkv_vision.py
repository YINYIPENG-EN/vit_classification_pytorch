import matplotlib
import matplotlib.pyplot as plt
plt.ion()


def qkv_view(feature_maps, num_rows=8, num_cols=8, head_idx=0):
    # feature_maps = [1,12,197,64] 【batch_size, num_heads, 197, C // num_heads】
    feature_maps = feature_maps[0, head_idx, 1:, :].cpu()  # 取出head中head_idx的特征序列 [196,64]
    feature_maps = feature_maps.reshape(14, 14, -1)
    feature_maps = feature_maps.permute(2, 0, 1)
    # 调整图像尺寸
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    # 遍历特征图并在每个子图中显示
    for i, ax in enumerate(axes.flat):
        # 获取第i个特征图
        feature_map = feature_maps[i]

        # 显示特征图
        ax.imshow(feature_map, cmap='hot')
        ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 调整子图布局
    plt.show()