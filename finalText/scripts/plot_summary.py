import matplotlib as mp
import matplotlib.pyplot as plt

import os
import glob
import csv
import numpy as np


path = './summary/'
extension = 'csv'
# os.chdir(path)
summaryFiles = [i for i in glob.glob(path + '*.{}'.format(extension))]

font_size = 20
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
fig.suptitle("Comparison between different CNN", fontsize=font_size + 10)

legend = []
for i in range(len(summaryFiles)):
    fileName = summaryFiles[i]
    splitName = fileName.split('.')
    legend.append('_'.join(splitName[2:(len(splitName) - 1)]))
    print("plot " + fileName)
    with open(fileName, "r") as fi:
        accuracy = []
        auc = []
        reader = csv.reader(fi, delimiter=',', skipinitialspace=True)
        for row in reader:
            accuracy.append(row[1])
            auc.append(row[2])

        max_points = [i + 1 for i,
                      j in enumerate(accuracy) if j == max(accuracy)]

        ax1.set_title('Accuracy', fontsize=font_size)
        ax1.set_xlabel("Feature height", fontsize=font_size)
        # ax1.set_yticks([])
        ax1.set_xticks((np.arange(1, len(accuracy) + 1, 1.0)), minor=False)
        p = ax1.plot(range(1, len(accuracy) + 1), accuracy)
        ax1.plot(max_points, np.repeat(max(accuracy), len(max_points)), 'o',
                 color=p[0].get_color(), label='_nolegend_')

        max_points = [i + 1 for i,
                      j in enumerate(auc) if j == max(auc)]
        ax2.set_title('AUROC', fontsize=font_size)
        ax2.set_xlabel("Feature height", fontsize=font_size)
        ax2.set_xticks((np.arange(1, len(auc) + 1, 1.0)), minor=False)
        p = ax2.plot(range(1, len(auc) + 1), auc)
        ax2.plot(max_points, np.repeat(max(auc), len(max_points)), 'o',
                 color=p[0].get_color(), label='_nolegend_')

ax1.legend(legend)
ax2.legend(legend)


plt.savefig("summary.pdf", bbox_inches='tight')
plt.close()
