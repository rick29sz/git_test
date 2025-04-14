import matplotlib.pyplot as plt
import numpy as np
#222222222
#  1111111111111111
def plot_losses(train_losses, name='', path=''):
    # 创建绘图
    plt.figure(figsize=(12, 7))

    plt.plot(train_losses, 
            label='Training Loss', 
            color='royalblue', 
            linewidth=2,
            # marker='o',
            markersize=8,
            markerfacecolor='red')

    # 美化图形
    plt.title('Training Loss Curve', fontsize=16, pad=20)
    plt.xlabel('iter', fontsize=14, labelpad=10)
    plt.ylabel('Loss Value', fontsize=14, labelpad=10)
    plt.legend(fontsize=12, framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 自动调整布局
    plt.tight_layout()

    # 保存图片（可选）
    plt.savefig(f'./{path}/{name}_loss.png', dpi=300, bbox_inches='tight')

# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    plot_losses([1,2,3,5,6,-1])
