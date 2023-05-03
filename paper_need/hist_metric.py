import numpy as np
import matplotlib.pyplot as plt

n_groups = 5

means_men = (7.78, 6.33, 7.56, 7.61, 7.78)
std_men = (0.42, 0.88, 0.60, 0.95, 0.71)

means_women = (7.56, 7.11, 7.83, 7.78, 7.94)
std_women = (1.07, 0.73, 1.07, 0.85, 0.85)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25

opacity = 1
error_config = {'ecolor': '0.3', 'capsize': 5}  # 添加 'capsize' 参数以显示“工”字形误差线

rects1 = ax.bar(index, means_men, bar_width,
                alpha=opacity, color='royalblue',
                yerr=std_men, error_kw=error_config,
                label='Typical')

rects2 = ax.bar(index + bar_width, means_women, bar_width,
                alpha=opacity, color='orange',
                yerr=std_women, error_kw=error_config,
                label='Untypical')

ax.set_xlabel('Evaluation item',  fontdict={'fontsize': 12, 'fontfamily': 'Times New Roman'})
ax.set_ylabel('Score',  fontdict={'fontsize': 12,  'fontfamily': 'Times New Roman'})
ax.set_title('Average scores corresponding to different metrics', fontdict={'fontsize': 13, 'fontweight': 'bold', 'fontfamily': 'Times New Roman'})
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('1', '2', '3', '4', '5'),  fontdict={'fontsize': 12,  'fontfamily': 'Times New Roman'})
ax.legend( prop={'size': 12,  'family': 'Times New Roman'})

# fig.tight_layout()
plt.ylim(0, 12)
plt.show()
