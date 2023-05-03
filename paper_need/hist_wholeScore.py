import numpy as np
import matplotlib.pyplot as plt

n_groups = 2

means_men = (37.06, 37.06)
std_men = (1.96, 1.58)

means_women = (38.22, 38.22)
std_women = (3.10, 2.62)

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

ax.set_xlabel('Analytical object',  fontdict={'fontsize': 12, 'fontfamily': 'Times New Roman'})
ax.set_ylabel('Score',  fontdict={'fontsize': 12,  'fontfamily': 'Times New Roman'})
ax.set_title('Average scores under different scenarios', fontdict={'fontsize': 13, 'fontweight': 'bold', 'fontfamily': 'Times New Roman'})
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('All cases', 'All layouts'),  fontdict={'fontsize': 12,  'fontfamily': 'Times New Roman'})
ax.legend( prop={'size': 12,  'family': 'Times New Roman'})

# fig.tight_layout()
plt.ylim(0, 52)
plt.show()
