import numpy as np
import matplotlib.pyplot as plt
import random
import glob

def parse_score(filepath):
    with open(filepath, 'r') as f:
        res = f.readlines()

    record = {}
    record['recall'] = []
    record['precision'] = []
    record['fscore'] = []

    for l in range(len(res)):
        if l+1 >= len(res):
            break
        if res[l][-2] == '=' and res[l+1][55:60] == 'ptime':
            # start from (l+5)
            content = res[l+5]

            record['recall'].append(float(content[33:39]))
            record['precision'].append(float(content[63:69]))
            record['fscore'].append(float(content[78:-1]))
    return record

def plot_compare(res, title):
    n_compare = len(res)

    new_level = ['750', '1500', '2000', '2500', '2750']
    xaxis = np.arange(5)
    color = ['red', 'blue', 'green', 'grey', 'orange']

    # Recall
    for i in range(len(res)):
        plt.plot(xaxis, res[i]['recall'], color=color[i], linestyle="-", linewidth="2", markersize="16", marker=".", label=title[i])

    plt.xticks(range(5), labels=new_level)
    plt.legend(loc='lower left')
    plt.title('Recall')
    plt.savefig('Recall.png')
    plt.clf()

    # Precision
    for i in range(len(res)):
        plt.plot(xaxis, res[i]['precision'], color=color[i], linestyle="-", linewidth="2", markersize="16", marker=".", label=title[i])

    plt.xticks(range(5), labels=new_level)
    plt.legend(loc='lower left')
    plt.title('Precision')
    plt.savefig('Precision.png')
    plt.clf()

    # F1-score
    for i in range(len(res)):
        plt.plot(xaxis, res[i]['fscore'], color=color[i], linestyle="-", linewidth="2", markersize="16", marker=".", label=title[i])

    plt.xticks(range(5), labels=new_level)
    plt.legend(loc='lower left')
    plt.title('F1-score')
    plt.savefig('F1score.png')
    plt.clf()

def plot_bar(res, label, idx):
    ptime = [750, 1500, 2000, 2500, 2750]
    precision, recall, fscore = [], [], []
    width = 0.25
    shift = 5
    for r in res:
        precision.append(r['precision'][idx])
        recall.append(r['recall'][idx])
        fscore.append(r['fscore'][idx])
   
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(label))*shift, recall, color='r', width=0.2, align='center', label='Recall', tick_label=label)
    plt.bar(np.arange(len(label))*shift-width, precision, color='b', width=0.2, align='center', label='Precision')
    plt.bar(np.arange(len(label))*shift+width, fscore, color='y', width=0.2, align='center', label='Fscore')
    plt.hlines(y=1.0, xmin=-0.5, xmax=20.5, linewidth=2, color='grey', linestyles='dashed')
    plt.legend(loc='lower right', prop={'size': 8})
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.savefig(f"{ptime[idx]}_compare.png")

if __name__ == '__main__':
    basedir = './results'

    dataset_opt = input('Dataset_opt: ')
    level = input('Level: ')

    res = []
    title = []
    while True:
        inp = input('Input the save_path to compare: (0: end)')
        title_opt = input('What name do you want?')

        if inp == '0' or len(res) == 5:
            print('Start plotting...')
            break

        filepath = glob.glob(f"{basedir}/{inp}/allTest_{dataset_opt}/*.log")[0]

        res.append(parse_score(filepath))
        title.append(title_opt)
        
    ptime = [750, 1500, 2000, 2500, 2750]
    for p in range(len(ptime)):
        plot_bar(res, title, p)






