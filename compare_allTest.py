import numpy as np
import matplotlib.pyplot as plt
import random

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

        filepath = f"{basedir}/{inp}/allTest_{dataset_opt}/threshold_allCase_testing_{level}.log"

        res.append(parse_score(filepath))
        title.append(title_opt)
        
    plot_compare(res, title)






