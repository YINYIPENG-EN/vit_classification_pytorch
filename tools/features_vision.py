import matplotlib
import matplotlib.pyplot as plt
plt.ion()

def Feaftures_vision(features_map, channel_index):
    B, N, C = features_map.shape
    features_map = features_map.reshape(1, 14, 14, C)
    features_map = features_map.permute(0, 3, 1, 2)
    plt.imshow(features_map[0, channel_index, :, :].cpu())