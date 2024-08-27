import numpy
from agents import load_nav_graph
import json
from sklearn.metrics import confusion_matrix
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import pickle
from sklearn.inspection import permutation_importance

with open('scan_to_node_to_room.pkl', 'rb') as f:
    SCAN_TO_NODE_TO_ROOM = pickle.load(f)

def nodeCoverageInTimeRange(dict_, scan, start, end):

    G = load_nav_graph(scan)

    total_nodes = len(G.nodes())

    unique_nodes_visited = list(set([dict_[time]['node'] for time in range(start, end+1)]))

    return unique_nodes_visited, len(unique_nodes_visited), total_nodes, unique_nodes_visited/total_nodes


def plotRoomDistribution(behavior = 'aggressive', scan = 'zsNo4HB9uLZ', id = 0, method = 'yolo'):

    with open(f'observation_dictionaries/{method}/zsNo4HB9uLZ/{behavior}/{id}_id.json', 'r') as file:
        dict_ = json.load(file)

    all_rooms = set([SCAN_TO_NODE_TO_ROOM[scan][node] for node in SCAN_TO_NODE_TO_ROOM[scan] ])
    print(all_rooms)
    all_rooms_count = defaultdict(int)
    #rooms = [SCAN_TO_NODE_TO_ROOM[scan][dict_[time]['node']] for time in dict_]

    for time in dict_:
        all_rooms_count[SCAN_TO_NODE_TO_ROOM[scan][dict_[time]['node']]] += 1
    labels, counts = [], []
    for room in all_rooms:
        labels.append(room)
        counts.append(all_rooms_count[room])
    # counter = Counter(rooms)

    # Extract the keys (unique strings) and their counts
    # labels, counts = zip(*counter.items())

    # Create the histogram
    plt.figure(figsize=(10, 6))
    # plt.bar(labels, counts, color='skyblue')
    plt.bar(labels, counts)
    plt.xlabel('Rooms')
    plt.ylabel('Visitation')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{behavior} ID: {id} {method} Room Visitation')
    plt.show()


def plotCoverage(dict_, method = 'yolo', scan = 'zsNo4HB9uLZ', fm = 0):
    
    plt.bar(dict_.keys(), dict_.values())

    if fm:
        plt.title(f'{method.upper()} with Compressed Memory')
    else:
        plt.title(f'{method.upper()} with Full Memory')
    plt.xlabel('Behavior')
    plt.ylabel(f'Coverage')

    plt.show()
    plt.savefig(f'answers/{method}/{scan}/coverage_full_mem_{fm}.png')

def getAnswerStats(answers, predicted):

    tn, fp, fn, tp = confusion_matrix(answers, predicted).ravel()

    acc = (tp + tn)/len(predicted)

    tpr = tp/(tp + fp)
    fpr = 1-tpr
    tnr = tn/(tn + fn)
    fnr = 1-tnr

    #return {'Accuracy': acc, 'TPR': tpr, 'FPR': fpr, 'TNR': tnr, 'FNR': fnr }
    return acc,  tpr,  fpr,  tnr,  fnr 

def plotvals(data, val = 'Accuracy', method = 'yolo', scan = 'zsNo4HB9uLZ', fm = 0):

    labels = list(data.keys())
    labels.sort()
    averages = [np.mean(data[l][val]) for l in labels]
    std_devs = [np.std(data[l][val]) for l in labels]
    plt.bar(labels, averages, yerr = std_devs, capsize = 15)

    if not fm:
        plt.title(f'{method.upper()} with Compressed Memory')
    else:
        plt.title(f'{method.upper()} with Full Memory')
    plt.xlabel('Behavior')
    plt.ylabel(f'{val}')

    # Show the plot
    plt.show()
    plt.savefig(f'answers/{method}/{scan}/{val}_full_mem_{fm}.png')


def map_response(response):
    response = response.lower()
    if 'no' in response:
        return 0
    elif 'yes' in response:
        return 1
    else:
        return None
    

def getPFI(model, X_val, y_val):

    return permutation_importance(model, X_val, y_val,
                            n_repeats=30,
                            random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{diabetes.feature_names[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-LLM behavior')

  
    
    parser.add_argument('--method',  type=str, default= 'yolo', metavar='N',
                        help='method')

    parser.add_argument('--scenario', '-s', type=int, default= 0, metavar='N',
                    help='scenario for subordinate agent') 
    
    parser.add_argument('--SCAN_ID', '-scan', type=str, default= 'zsNo4HB9uLZ', metavar='N',
                    help='scan_ID') 
    
    parser.add_argument('--value', '-v', type=str, default= 'Accuracy', metavar='N',
                    help='value') 
    
    parser.add_argument('--full_mem', '-f', type=int, default= 0, metavar='N',
                    help='full or compressed memory') 
    
    parser.add_argument('--agent_id', '-a', type=int, default= 0, metavar='N',
                    help='agent id') 
    
    parser.add_argument('--behavior', '-b', type=str, default= 'aggressive', metavar='N',
                        help='exploration behavior')
    
    args = parser.parse_args() 

    
    df = pd.read_csv(f'answers_fixed_exp_no_common_sense/{args.method}/{args.SCAN_ID}/answers_0_with_obs_full_mem_{args.full_mem}.csv')

    col = "['dining room', 'kitchen']_response" #f'{args.behavior}_helpful_{args.agent_id}_response'
    #col = "['bathroom', 'bedroom']_response"
    df[col] = df[col].apply(map_response)

    def calculate_tpr_fnr(group):
        tp = ((group['Answer'] == 1) & (group[col] == 1)).sum()
        fn = ((group['Answer'] == 1) & (group[col] == 0)).sum()
        total_positive = tp + fn
        tpr = tp / total_positive if total_positive > 0 else 0
        fnr = fn / total_positive if total_positive > 0 else 0
        return pd.Series({'TPR': tpr, 'FNR': fnr})

    accuracy_per_room = df.groupby('Room').apply(lambda x: (x['Answer'] == x[col]).mean())

    #plt.figure(figsize=(10, 6))
    accuracy_per_room.plot(kind='bar', color='skyblue')
    plt.xlabel('Room')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Room')
    plt.xticks(rotation=45, ha='right')  # Tilt labels and align to the right
    plt.tight_layout()  # Adjust layout to make room for the labels
    plt.show()

    tpr_fnr_per_room = df.groupby('Room').apply(calculate_tpr_fnr)

    # Plot TPR and FNR
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    tpr_fnr_per_room['TPR'].plot(kind='bar', color='skyblue', ax=ax[0])
    ax[0].set_xlabel('Room')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('True Positive Rate per Room')
    ax[0].set_xticklabels(tpr_fnr_per_room.index, rotation=45, ha='right')

    tpr_fnr_per_room['FNR'].plot(kind='bar', color='lightcoral', ax=ax[1])
    ax[1].set_xlabel('Room')
    ax[1].set_ylabel('False Negative Rate')
    ax[1].set_title('False Negative Rate per Room')
    ax[1].set_xticklabels(tpr_fnr_per_room.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


    if True:
        pass
        #df = pd.read_csv(f'answers/{args.method}/{args.SCAN_ID}/answers_0_with_obs_full_mem_{args.full_mem}.csv')
        df = pd.read_csv(f'answers_fixed_exp_no_common_sense/{args.method}/{args.SCAN_ID}/answers_0_with_obs_full_mem_{args.full_mem}.csv')

        answers_list = df['Answer'].tolist()



        # Apply the function to any column with 'response' in the name
        response_columns = [col for col in df.columns if 'response' in col.lower()]
        data = {}
        
        for col in response_columns:
            
            if col.split('_')[0] not in data:
                data[col.split('_')[0]] = {}
                data[col.split('_')[0]]['Accuracy'] = []
                data[col.split('_')[0]]['TPR'] = []
                data[col.split('_')[0]]['FPR'] = []
                data[col.split('_')[0]]['TNR'] = []
                data[col.split('_')[0]]['FNR'] = []


            vals = df[col].tolist()
            vals = [0 if 'no' in val.lower() else 1 for val in vals]

            acc, tpr, fpr, tnr, fnr = getAnswerStats(answers_list, vals)
            #data[col.split('_')[0]] = getAnswerStats(answers_list, vals)
            data[col.split('_')[0]]['Accuracy'].append(acc)
            data[col.split('_')[0]]['TPR'].append(tpr)
            data[col.split('_')[0]]['FPR'].append(fpr)
            data[col.split('_')[0]]['TNR'].append(tnr)
            data[col.split('_')[0]]['FNR'].append(fnr)

            
            #print(col, getAnswerStats(answers_list, vals))
        print(data)
        plotvals(data, val= args.value, method = args.method, fm = args.full_mem)

    else:
        pass
        plotRoomDistribution(behavior = args.behavior, id = args.agent_id, method = args.method)

    # for col in response_columns:

