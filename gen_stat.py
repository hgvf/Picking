import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import argparse
import bisect

def gen_intensity_snr_Stat_csv(case_stat, snr_level, intensity_level, output_dir, test=False, plot=True):
    res = {}
    for i in intensity_level:
        res[str(i)] = {}
        
        for j in range(len(snr_level)):
            res[str(i)][str(j)] = []

    for i in range(len(case_stat['res'])):
        tmp_inten = int(case_stat['intensity'][i])
        res[str(intensity_level[tmp_inten])][case_stat['snr'][i]].append(case_stat['res'][i])

    inten_res = {}
    for k, v in res.items():
        inten_res[k] = {}
        t_tp, t_fp, t_tn, t_fn = 0, 0, 0, 0

        for snr_k, snr_v in v.items():
            inten_res[k][snr_k] = {}

            tp, fp, tn, fn = snr_v.count('tp'), snr_v.count('fp'), snr_v.count('tn'), snr_v.count('fn')
            t_tp += tp
            t_fp += fp
            t_tn += tn
            t_fn += fn

            inten_res[k][snr_k]['TP'], inten_res[k][snr_k]['FP'], inten_res[k][snr_k]['TN'], inten_res[k][snr_k]['FN'] = tp, fp, tn, fn
            
            p = tp / (tp+fp) if (tp+fp) != 0 else 0
            r = tp / (tp+fn) if (tp+fn) != 0 else 0
            f = 2*p*r / (p+r) if p+r != 0 else 0
            
            inten_res[k][snr_k]['count'], inten_res[k][snr_k]['precision'], inten_res[k][snr_k]['recall'], inten_res[k][snr_k]['fscore'] = tp+fp+tn+fn, p, r, f

            if inten_res[k][snr_k]['count'] != 0 and f == 0.0:
                inten_res[k][snr_k]['fscore'] = 0.3

        inten_res[k]['total'] = {}
        inten_res[k]['total']['TP'], inten_res[k]['total']['FP'], inten_res[k]['total']['TN'], inten_res[k]['total']['FN'] = t_tp, t_fp, t_tn, t_fn
        inten_res[k]['total']['count'] = int(t_tp + t_fp + t_tn + t_fn)
        inten_res[k]['total']['precision'] = t_tp / (t_tp+t_fp) if (t_tp+t_fp) != 0 else 0
        inten_res[k]['total']['recall'] = t_tp / (t_tp+t_fn) if (t_tp+t_fn) != 0 else 0
        inten_res[k]['total']['fscore'] = 2*inten_res[k]['total']['precision']*inten_res[k]['total']['recall'] / (inten_res[k]['total']['precision']+inten_res[k]['total']['recall']) if (inten_res[k]['total']['precision']+inten_res[k]['total']['recall']) != 0 else 0    

    if plot:
        df = pd.DataFrame.from_dict(inten_res)

        if not test:
            filepath = 'intensity_snr_stat.csv'
        else:
            filepath = 'test_intensity_snr_stat.csv'

        df.to_csv(os.path.join(output_dir, filepath))

    return inten_res

def gen_caseStat_4axis_csv(case_stat, snr_level, intensity_level, output_dir, test=False, plot=True):
    t_tp, t_fp, t_tn, t_fn = 0, 0, 0, 0
    recall, precision, fscore, count = [], [], [], []
    axis1, axis2, axis3, axis4 = [], [], [], []
    axis = {}

    for i in range(1, 5):
        axis[str(i)] = {}
        axis[str(i)]['precision'] = {}
        axis[str(i)]['recall'] = {}
        axis[str(i)]['fscore'] = {}
        axis[str(i)]['count'] = {}

    for i in range(len(case_stat['res'])):
        if float(case_stat['snr'][i]) >= 3 and float(case_stat['intensity'][i]) >= 4:
            axis1.append(case_stat['res'][i])
            
        elif float(case_stat['snr'][i]) >= 3 and float(case_stat['intensity'][i]) < 4:
            axis2.append(case_stat['res'][i])
            
        elif float(case_stat['snr'][i]) < 3 and float(case_stat['intensity'][i]) < 4:
            axis3.append(case_stat['res'][i])
            
        elif float(case_stat['snr'][i]) < 3 and float(case_stat['intensity'][i]) >= 4:
            axis4.append(case_stat['res'][i])

    all_axis = [axis1, axis2, axis3, axis4]
    for i, a in enumerate(all_axis):
        tp = a.count('tp')
        fp = a.count('fp')
        tn = a.count('tn')
        fn = a.count('fn')
        
        axis[str(i+1)]['TP'] = tp
        axis[str(i+1)]['FP'] = fp
        axis[str(i+1)]['TN'] = tn
        axis[str(i+1)]['FN'] = fn

        p = tp / (tp+fp) if (tp+fp) != 0 else 0
        r = tp / (tp+fn) if (tp+fn) != 0 else 0
        f = 2*p*r / (p+r) if p+r != 0 else 0
        
        axis[str(i+1)]['count'] = tp+fp+tn+fn
        axis[str(i+1)]['recall'] = r
        axis[str(i+1)]['precision'] = p
        axis[str(i+1)]['fscore'] = f

    if plot:
        df = pd.DataFrame.from_dict(axis)
        
        if not test:
            filepath = '4axis_stat.csv'
        else:
            filepath = '4axis_stat.csv'

        df.to_csv(os.path.join(output_dir, filepath))

def gen_snrStat_csv(snr_stat, snr_level, output_dir, test=False, plot=True):
    t_tp, t_fp, t_tn, t_fn = 0, 0, 0, 0
    
    snr_score = {}
    for k in range(len(snr_stat.keys())):
        res = snr_stat[str(k)]

        tp = res.count('tp')
        fp = res.count('fp')
        tn = res.count('tn')
        fn = res.count('fn')

        t_tp += tp
        t_fp += fp
        t_tn += tn
        t_fn += fn
        
        if k == 0:
            k = '< ' + str(snr_level[1])
        else:
            k = '>= ' + str(snr_level[k])
        
        snr_score[k] = {}
        snr_score[k]['TP'] = tp
        snr_score[k]['FP'] = fp
        snr_score[k]['TN'] = tn
        snr_score[k]['FN'] = fn

        snr_score[k]['count'] = int(tp + fp + tn + fn)
        snr_score[k]['precision'] = tp / (tp+fp) if (tp+fp) != 0 else 0
        snr_score[k]['recall'] = tp / (tp+fn) if (tp+fn) != 0 else 0
        snr_score[k]['fscore'] = 2*snr_score[k]['precision']*snr_score[k]['recall'] / (snr_score[k]['precision']+snr_score[k]['recall']) if (snr_score[k]['precision']+snr_score[k]['recall']) != 0 else 0    

    snr_score['total'] = {}
    snr_score['total']['TP'], snr_score['total']['FP'], snr_score['total']['TN'], snr_score['total']['FN'] = t_tp, t_fp, t_tn, t_fn
    snr_score['total']['count'] = int(t_tp + t_fp + t_tn + t_fn)
    snr_score['total']['precision'] = t_tp / (t_tp+t_fp) if (t_tp+t_fp) != 0 else 0
    snr_score['total']['recall'] = t_tp / (t_tp+t_fn) if (t_tp+t_fn) != 0 else 0
    snr_score['total']['fscore'] = 2*snr_score['total']['precision']*snr_score['total']['recall'] / (snr_score['total']['precision']+snr_score['total']['recall']) if (snr_score['total']['precision']+snr_score['total']['recall']) != 0 else 0    
    
    if plot:
        df = pd.DataFrame.from_dict(snr_score)
        
        if not test:
            filepath = 'snr_stat.csv'
        else:
            filepath = 'test_snr_stat.csv'

        df.to_csv(os.path.join(output_dir, filepath))

    return snr_score

def gen_intensityStat_csv(intensity_stat, intensity_level, output_dir, test=False, plot=True):
    t_tp, t_fp, t_tn, t_fn = 0, 0, 0, 0

    intensity_score = {}
    for k in range(len(intensity_stat.keys())):
        res = intensity_stat[str(k)]

        tp = res.count('tp')
        fp = res.count('fp')
        tn = res.count('tn')
        fn = res.count('fn')

        t_tp += tp
        t_fp += fp
        t_tn += tn
        t_fn += fn
        
        k = str(intensity_level[k])
        if k == '5.5' or k == '6.5':
            k = k[0] + ' strong'
        elif k == '5' or k == '6':
            k = k[0] + ' weak'
            
        intensity_score[k] = {}
        intensity_score[k]['TP'] = tp
        intensity_score[k]['FP'] = fp
        intensity_score[k]['TN'] = tn
        intensity_score[k]['FN'] = fn

        intensity_score[k]['count'] = int(tp + fp + tn + fn)
        intensity_score[k]['precision'] = tp / (tp+fp) if (tp+fp) != 0 else 0
        intensity_score[k]['recall'] = tp / (tp+fn) if (tp+fn) != 0 else 0
        intensity_score[k]['fscore'] = 2*intensity_score[k]['precision']*intensity_score[k]['recall'] / (intensity_score[k]['precision']+intensity_score[k]['recall']) if (intensity_score[k]['precision']+intensity_score[k]['recall']) != 0 else 0    

        if intensity_score[k]['fscore'] == 0.0 and intensity_score[k]['count'] != 0:
            intensity_score[k]['fscore'] = 0.3

    intensity_score['total'] = {}
    intensity_score['total']['TP'], intensity_score['total']['FP'], intensity_score['total']['TN'], intensity_score['total']['FN'] = t_tp, t_fp, t_tn, t_fn
    intensity_score['total']['count'] = int(t_tp + t_fp + t_tn + t_fn)
    intensity_score['total']['precision'] = t_tp / (t_tp+t_fp) if (t_tp+t_fp) != 0 else 0
    intensity_score['total']['recall'] = t_tp / (t_tp+t_fn) if (t_tp+t_fn) != 0 else 0
    intensity_score['total']['fscore'] = 2*intensity_score['total']['precision']*intensity_score['total']['recall'] / (intensity_score['total']['precision']+intensity_score['total']['recall']) if (intensity_score['total']['precision']+intensity_score['total']['recall']) != 0 else 0    
    
    if plot:
        df = pd.DataFrame.from_dict(intensity_score)
        
        if not test:
            filepath = 'intensity_stat.csv'
        else:
            filepath = 'test_intensity_stat.csv'

        df.to_csv(os.path.join(output_dir, filepath))

    return intensity_score

def gen_boxShape(data, title, output_dir):
    plt.title(title + '   (Prediction - Ground truth)')
    plt.boxplot(data/100)
    plt.ylabel('sample (prediction - ground_truth)')
    plt.xlabel('Model')
    plt.savefig(os.path.join(output_dir, './' + title + '_box.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def gen_hist(data, title, total_number, output_dir, sampling_rate=100):
    data = data / sampling_rate
    outlier_ratio = round(1 - data.shape[0] / total_number, 2)

    plt.title(title + '   (Prediction - Ground truth)')
    plt.hist(data, edgecolor='black')
    plt.ylabel('counts')
    plt.xlabel('Time residual (s)')
    plt.axvline(np.mean(data), label='mean', color='salmon', linestyle='--')
    plt.axvline(np.median(data), label='median', color='burlywood', linestyle='--')
    plt.text(0.95, 0.5, 'Outlier: '+str(outlier_ratio), horizontalalignment='right',
     verticalalignment='center', transform = plt.gca().transAxes)
    plt.legend()

    plt.savefig(os.path.join(output_dir, './' + title + '_hist.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def parse_score_single(res):
    record = {}
    for l in range(len(res)):
        if res[l][-2] == '=':
            text = res[l+1].split('|')[-1]
            content = text.split(':')

            if content[0].strip() == 'Mode':
                if content[1].strip() == 'single':
                    for i in range(3, len(res), 8):
                        if i+l >= len(res):
                            break
                        prob = res[l+i].split(':')[-1].strip()
                        record[prob] = res[l+i+2:l+i+8]

                        if float(prob) == 0.85:
                            break
    
    prob = []
    recall = []
    precision = []
    fscore = []
    for k in record.keys():    
        tmp = record[k][0].split('|')[-1].split(',')
        for t in tmp:
            title = t.split('=')[0].strip()

            if title == 'FPR':
                continue
            elif title == 'TPR':
                prob.append(float(k))
                recall.append(float(t.split('=')[1].strip()))
            elif title == 'Precision':
                precision.append(float(t.split('=')[1].strip()))
            elif title == 'Fscore':
                fscore.append(float(t.split('=')[1].strip()))
                
    return prob, recall, precision, fscore

def parse_score(res, mode):
    record = {}
    for l in range(len(res)):
        if res[l][-2] == '=':
            text = res[l+1].split('|')[-1]
            content = text.split(':')

            if content[0].strip() == 'Mode':
                if content[1].strip() == mode:
                    for i in range(3, len(res), 8):
                        if i+l >= len(res):
                            break

                        prob = res[l+i].split('|')[-1].split(':')[-1].strip()
                        trigger = res[l+i+1].split('|')[-1].split(':')[-1].strip()

                        if str(prob) not in record.keys():
                            record[prob] = {}

                        record[prob][trigger] = res[l+i+2]

                        if float(prob) == 0.85:
                            break

    prob = []
    trigger = []
    recall = []
    precision = []
    fscore = []
    for k in record.keys():    
        max_fscore, max_recall, max_precision = 0.0, 0.0, 0.0
        best_tri = 0

        for tri in record[k].keys():
            tmp = record[k][tri].split('|')[-1].split(',')

            for t in tmp:
                title = t.split('=')[0].strip()

                if title == 'FPR':
                    continue
                elif title == 'TPR':
                    tmp_recall = float(t.split('=')[1].strip())
                elif title == 'Precision':
                    tmp_precision = float(t.split('=')[1].strip())
                elif title == 'Fscore':
                    tmp_fscore = float(t.split('=')[1].strip())

            if tmp_fscore > max_fscore:
                max_fscore = tmp_fscore
                max_recall = tmp_recall
                max_precision = tmp_precision
                best_tri = tri

        prob.append(float(k))
        trigger.append(int(best_tri))
        recall.append(max_recall)
        precision.append(max_precision)
        fscore.append(max_fscore)

    return prob, trigger, recall, precision, fscore

def plot_roc_recall_precision(data1, data2, prob, title, output_dir, trigger=None):
    plt.title(title)
    plt.plot(data1, data2)
    plt.scatter(data1, data2, label='Threshold (prob, sample)')
    plt.legend(loc = 'lower right')
    plt.xlim([np.min(recall)-0.01, 1])
    plt.ylim([np.min(precision)-0.01, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')

    if trigger is not None:
        for i in range(len(recall)):
            # plt.annotate('(' + str(prob[i]) + ',' + str(trigger[i]) + ')', (recall[i]-0.08, precision[i]-0.035))
            plt.annotate('(' + str(prob[i]) + ',' + str(trigger[i]) + ')', (recall[i], precision[i]))
    else:
        for i in range(len(precision)):
            # plt.annotate(prob[i], (recall[i]-0.08, precision[i]-0.035))
            plt.annotate(prob[i], (recall[i], precision[i]))
    plt.savefig(os.path.join(output_dir, './' + title + '_recall_precision_roc.png'), dpi=300)
    plt.clf()
    plt.close()

def plot_roc_fscore(fscore, prob, title, output_dir, trigger=None):
    plt.title(title)
    plt.plot(prob, fscore)
    plt.scatter(prob, fscore, label='Threshold (prob, sample)')
    plt.legend(loc = 'lower right')
    plt.xlim([np.min(prob), np.max(prob)])
    plt.ylim([np.min(fscore)-0.01, 1])
    plt.ylabel('Fscore')
    plt.xlabel('Threshold')

    if trigger is not None:
        for i in range(len(fscore)):
            # plt.annotate('(' + str(prob[i]) + ',' + str(trigger[i]) + ')', (prob[i]+0.01, fscore[i]-0.08))
            plt.annotate('(' + str(prob[i]) + ',' + str(trigger[i]) + ')', (prob[i], fscore[i]))
    else:
        for i in range(len(fscore)):
            # plt.annotate(prob[i], (prob[i]+0.01, fscore[i]-0.08))
            plt.annotate(prob[i], (prob[i], fscore[i]))
    plt.savefig(os.path.join(output_dir, './' + title + '_fscore_roc.png'), dpi=300)
    plt.clf()
    plt.close()

def plot_snr_bar(snr_score, title, output_dir):
    fscore = []
    recall = []
    precision = []
    label = []
    for idx, k in enumerate(snr_score.keys()):
        if idx == 0:
            continue

        fscore.append(snr_score[k]['fscore'])
        recall.append(snr_score[k]['recall'])
        precision.append(snr_score[k]['precision'])
        
        label.append(k)

    width = 0.25

    plt.figure(figsize=(25, 5))
    plt.bar(label, recall, color='r', width=0.2, align='center', label='Recall')
    plt.bar(np.arange(len(label))-width, precision, color='b', width=0.2, align='center', label='Precision')
    plt.bar(np.arange(len(label))+width, fscore, color='y', width=0.2, align='center', label='Fscore')
    plt.hlines(y=1.0, xmin=-0.5, xmax=20.5, linewidth=2, color='grey', linestyles='dashed')
    plt.legend(loc='lower right', prop={'size': 15})
    plt.ylabel('Score')
    plt.xlabel('SNR')

    plt.savefig(os.path.join(output_dir, './' + title + '_snr_bar.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_intensity_bar(intensity_score, title, output_dir):
    fscore = []
    recall = []
    precision = []
    label = []
    for idx, k in enumerate(intensity_score.keys()):
        fscore.append(intensity_score[k]['fscore'])
        recall.append(intensity_score[k]['recall'])
        precision.append(intensity_score[k]['precision'])
        
        label.append(k)

    width = 0.25

    plt.figure(figsize=(18, 5))
    plt.bar(label, recall, color='r', width=0.2, align='center', label='Recall')
    plt.bar(np.arange(len(label))-width, precision, color='b', width=0.2, align='center', label='Precision')
    plt.bar(np.arange(len(label))+width, fscore, color='y', width=0.2, align='center', label='Fscore')
    plt.hlines(y=1.0, xmin=-0.5, xmax=18.5, linewidth=2, color='grey', linestyles='dashed')
    plt.legend(loc='lower right')
    plt.ylabel('Score')
    plt.xlabel('Intensity')

    plt.savefig(os.path.join(output_dir, './' + title + '_intensity_bar.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    
def plot_intensity_snr_bar(res, snr_level, title, output_dir):

    for idx, k in enumerate(res.keys()):
        fscore = []
        recall = []
        precision = []
        label = []
        for jdx, j in enumerate(res[k].keys()):
            fscore.append(res[k][j]['fscore'])
            recall.append(res[k][j]['recall'])
            precision.append(res[k][j]['precision'])

            if j == '0':
                l = '< ' + str(snr_level[1])
            elif j == 'total':
                l = 'Total'
            else:
                l = '>= ' + str(snr_level[int(j)])

            label.append(l)
        
        width = 0.25
        
        plt.figure(figsize=(18, 5))
        plt.bar(label, recall, color='r', width=0.2, align='center', label='Recall')
        plt.bar(np.arange(len(label))-width, precision, color='b', width=0.2, align='center', label='Precision')
        plt.bar(np.arange(len(label))+width, fscore, color='y', width=0.2, align='center', label='Fscore')
        plt.hlines(y=1.0, xmin=-0.5, xmax=18.5, linewidth=2, color='grey', linestyles='dashed')
        plt.legend(loc='lower right')
        plt.ylabel('Score')
        plt.xlabel('SNR')
        
        if k == '5' or k == '6':
            k = k[0] + ' weak'
        elif k == '5.5' or k == '6.5':
            k = k[0] + ' strong'
            
        plot_title = 'Intensity: ' + k
        plt.title(plot_title)
        
        plt.savefig(os.path.join(output_dir, './' + title + '_intensity_' + str(k) + '_bar.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

def plot_fp_bar(diff, title, output_dir):
    diff = np.array(diff)
    plt.rcParams.update({'font.size': 8})
    max_diff, min_diff = np.max(diff), np.min(diff)
    left, right = min_diff // 100 + 1, max_diff // 100 + 2
    diff_level = list(np.arange(left*100, -100, 100)) + list(np.arange(-100, -40, 10)) + list(np.arange(50, 100, 10)) + list(np.arange(100, right*100, 100))

    stat = np.zeros(len(diff_level))
    for i in diff:
        idx = bisect.bisect_right(diff_level, i)
        
        stat[idx] += 1

    new_level = []
    for d in diff_level:
        new_level.append('< ' + str(d/100) + ' s')
        
    plt.figure(figsize=(18, 5))
    plt.bar(np.arange(len(diff_level)), stat)
    plt.xticks(range(len(stat)), labels=new_level)
    plt.ylabel('Counts')
    plt.xlabel('Time residual')
    plt.title(title + ' picking\'s error distribution from false positive')
        
    plt.savefig(os.path.join(output_dir, './' + title + '_fp_diff_bar.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

# for comparison
def comp_plot_snr_bar(snr_score1, snr_score2, model1, model2, level, title, output_dir):
    fscore = []
    recall = []
    precision = []
    label = []
    for idx, k in enumerate(snr_score1.keys()):
        if idx == 0:
            continue

        fscore.append(snr_score1[k]['fscore'] - snr_score2[k]['fscore'])
        recall.append(snr_score1[k]['recall'] - snr_score2[k]['recall'])
        precision.append(snr_score1[k]['precision'] - snr_score2[k]['precision'])
        
        label.append(k)

    width = 0.25

    # decide the range of y-axis
    max_y = max(max(abs(np.array(fscore))), max(abs(np.array(recall))), max(abs(np.array(precision))))

    plt.figure(figsize=(25, 5))
    plt.bar(label, recall, color='r', width=0.2, align='center', label='Recall')
    plt.bar(np.arange(len(label))-width, precision, color='b', width=0.2, align='center', label='Precision')
    plt.bar(np.arange(len(label))+width, fscore, color='y', width=0.2, align='center', label='Fscore')
    plt.hlines(y=0.0, xmin=-0.5, xmax=20.5, linewidth=2, color='grey', linestyles='dashed')
    plt.ylim([-max_y - 0.05, max_y + 0.05])
    plt.legend(loc='lower right', prop={'size': 15})
    plt.ylabel('Score difference')
    plt.xlabel('SNR')
    # plt.title(f"{model1} - {model2}")

    title += '_with_' + model2 + '_' + str(level)
    plt.savefig(os.path.join(output_dir, './' + title + '_snr_bar.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def comp_plot_intensity_bar(intensity_score1, intensity_score2, model1, model2, level, title, output_dir):
    fscore = []
    recall = []
    precision = []
    label = []
    for idx, k in enumerate(intensity_score1.keys()):
        fscore.append(intensity_score1[k]['fscore'] - intensity_score2[k]['fscore'])
        recall.append(intensity_score1[k]['recall'] - intensity_score2[k]['recall'])
        precision.append(intensity_score1[k]['precision'] - intensity_score2[k]['precision'])
        
        label.append(k)

    width = 0.25

    # decide the range of y-axis
    max_y = max(max(abs(np.array(fscore))), max(abs(np.array(recall))), max(abs(np.array(precision))))
    
    plt.figure(figsize=(18, 5))
    plt.bar(label, recall, color='r', width=0.2, align='center', label='Recall')
    plt.bar(np.arange(len(label))-width, precision, color='b', width=0.2, align='center', label='Precision')
    plt.bar(np.arange(len(label))+width, fscore, color='y', width=0.2, align='center', label='Fscore')
    plt.hlines(y=0.0, xmin=-0.5, xmax=18.5, linewidth=2, color='grey', linestyles='dashed')
    plt.ylim([-max_y - 0.05, max_y + 0.05])
    plt.legend(loc='lower right')
    plt.ylabel('Score difference')
    plt.xlabel('Intensity')
    plt.title(f"{model1} - {model2}")

    title += '_with_' + model2 + '_' + str(level)
    plt.savefig(os.path.join(output_dir, './' + title + '_intensity_bar.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument("--only_test", type=bool, default=False)
    parser.add_argument('--only_compare', type=bool, default=False)
    parser.add_argument('--compare', type=bool, default=False)
    parser.add_argument('--compare_save_path', type=str, default='tmp')
    parser.add_argument("--level", type=int, default=-1)
    parser.add_argument('--allTest', type=bool, default=False)
    parser.add_argument('--p_timestep', type=int)
    opt = parser.parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    stat_dir = os.path.join(output_dir, 'stat')
    print('saving path: ', stat_dir)
    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir)

    if not opt.allTest:
        if opt.level == -1:
            level = 'all'
        else:
            level = str(opt.level)

        output_dir = os.path.join(output_dir, level)
    else:
        output_dir = os.path.join(output_dir, 'allTest_taiwan')
        output_dir = os.path.join(output_dir, str(opt.p_timestep))

    # snr_level = list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    snr_level = [-9999] + list(np.arange(-1.0, 0.0, 0.5)) + list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    intensity_level = [-1, 0, 1, 2, 3, 4, 5, 5.5, 6, 6.5, 7]

    if not opt.only_test and not opt.only_compare:
        print('Loading dev prediction...')
        with open(os.path.join(output_dir, "snr_stat_"+str(opt.level)+".json")) as f:
            snr_stat = json.load(f)
        with open(os.path.join(output_dir, "intensity_stat_"+str(opt.level)+".json")) as f:
            intensity_stat = json.load(f)
        with open(os.path.join(output_dir, "case_stat_"+str(opt.level)+".json")) as f:
            case_stat = json.load(f)
        with open(os.path.join(output_dir, "diff_"+str(opt.level)+".pkl"), 'rb') as f:
            diff = pickle.load(f)
            diff = np.array(diff)
            total_diff = diff.shape[0]
            fp_diff = diff[np.logical_or(diff<-50, diff>50)]
            diff = diff[np.logical_and(diff>=-50, diff<=50)]
        with open(os.path.join(output_dir, "abs_diff_"+str(opt.level)+".pkl"), 'rb') as f:
            abs_diff = pickle.load(f)
            abs_diff = np.array(abs_diff)
            total_abs_diff = abs_diff.shape[0]
            abs_diff = abs_diff[abs_diff <= 50]

        if opt.level == -1:
            threshold_path = 'threshold_-1.log'
        else:
            threshold_path = 'threshold_' + str(opt.level) + '.log'
        with open(os.path.join(output_dir, threshold_path), 'r') as f:
            res = f.readlines()

        # generate statistical, box shape, and histogram
        print('Generating dev statistical...')
        snr_score = gen_snrStat_csv(snr_stat, snr_level, stat_dir)
        intensity_score = gen_intensityStat_csv(intensity_stat, intensity_level, stat_dir)
        gen_caseStat_4axis_csv(case_stat, snr_level, intensity_level, stat_dir)
        inten_res = gen_intensity_snr_Stat_csv(case_stat, snr_level, intensity_level, stat_dir)

        plot_snr_bar(snr_score, 'dev', stat_dir)
        plot_intensity_bar(intensity_score, 'dev', stat_dir)
        plot_intensity_snr_bar(inten_res, snr_level, 'dev', stat_dir)
        plot_fp_bar(fp_diff, 'dev', stat_dir)

        gen_boxShape(diff, 'diff', stat_dir)
        gen_boxShape(abs_diff, 'abs_diff', stat_dir)

        gen_hist(diff, 'diff', total_diff, stat_dir)
        gen_hist(abs_diff, 'abs_diff', total_abs_diff, stat_dir)

        # generate roc curve (single mode)
        print('Generating roc curve... (single)')
        prob, recall, precision, fscore = parse_score_single(res)
        plot_roc_recall_precision(recall, precision, prob, 'single', stat_dir)
        plot_roc_fscore(fscore, prob, 'single', stat_dir)

        # generate roc curve (avg mode)
        print('Generating roc curve... (avg)')
        prob, trigger, recall, precision, fscore = parse_score(res, 'avg')
        plot_roc_recall_precision(recall, precision, prob, 'avg', stat_dir, trigger)
        plot_roc_fscore(fscore, prob, 'avg', stat_dir, trigger)

        # generate roc curve (continue mode)
        print('Generating roc curve... (continue)')
        prob, trigger, recall, precision, fscore = parse_score(res, 'continue')
        plot_roc_recall_precision(recall, precision, prob, 'continue', stat_dir, trigger)
        plot_roc_fscore(fscore, prob, 'continue', stat_dir, trigger)

    # =========================== testing set =========================== #
    print('Loading test prediction...')
    with open(os.path.join(output_dir, "test_snr_stat_"+str(opt.level)+".json")) as f:
        snr_stat = json.load(f)
    with open(os.path.join(output_dir, "test_intensity_stat_"+str(opt.level)+".json")) as f:
        intensity_stat = json.load(f)
    with open(os.path.join(output_dir, "test_case_stat_"+str(opt.level)+".json")) as f:
        case_stat = json.load(f)
    with open(os.path.join(output_dir, "test_diff_"+str(opt.level)+".pkl"), 'rb') as f:
        diff = pickle.load(f)
        diff = np.array(diff)
        total_diff = diff.shape[0]
        fp_diff = diff[np.logical_or(diff<-50, diff>50)]
        diff = diff[np.logical_and(diff>=-50, diff<=50)]
    with open(os.path.join(output_dir, "test_abs_diff_"+str(opt.level)+".pkl"), 'rb') as f:
        abs_diff = pickle.load(f)
        abs_diff = np.array(abs_diff)
        total_abs_diff = abs_diff.shape[0]
        abs_diff = abs_diff[abs_diff <= 50]

    # generate statistical, box shape, and histogram
    print('Generating test statistical...')
    snr_score = gen_snrStat_csv(snr_stat, snr_level, stat_dir, True)
    intensity_score = gen_intensityStat_csv(intensity_stat, intensity_level, stat_dir, True)
    gen_caseStat_4axis_csv(case_stat, snr_level, intensity_level, stat_dir, True)
    inten_res = gen_intensity_snr_Stat_csv(case_stat, snr_level, intensity_level, stat_dir, True)

    if not opt.only_compare:
        plot_snr_bar(snr_score, 'test', stat_dir)
        plot_intensity_bar(intensity_score, 'test', stat_dir)
        plot_intensity_snr_bar(inten_res, snr_level, 'test', stat_dir)
        plot_fp_bar(fp_diff, 'test', stat_dir)

        gen_boxShape(diff, 'test_diff', stat_dir)
        gen_boxShape(abs_diff, 'test_abs_diff', stat_dir)

        gen_hist(diff, 'test_diff', total_diff, stat_dir)
        gen_hist(abs_diff, 'test_abs_diff', total_abs_diff, stat_dir)

    # starting compare
    if opt.compare:
        print(f'Loading {opt.compare_save_path} prediction...')
        output_dir = os.path.join('./results', opt.compare_save_path)
        output_dir = os.path.join(output_dir, level)

        with open(os.path.join(output_dir, "test_snr_stat_"+str(opt.level)+".json")) as f:
            snr_stat = json.load(f)
        with open(os.path.join(output_dir, "test_intensity_stat_"+str(opt.level)+".json")) as f:
            intensity_stat = json.load(f)
        with open(os.path.join(output_dir, "test_case_stat_"+str(opt.level)+".json")) as f:
            case_stat = json.load(f)
        
        c_snr_score = gen_snrStat_csv(snr_stat, snr_level, stat_dir, True, False)
        c_intensity_score = gen_intensityStat_csv(intensity_stat, intensity_level, stat_dir, True, False)

        comp_plot_snr_bar(snr_score, c_snr_score, opt.save_path, opt.compare_save_path, level, 'Compare', stat_dir)
        comp_plot_intensity_bar(intensity_score, c_intensity_score, opt.save_path, opt.compare_save_path, level, 'Compare', stat_dir)
