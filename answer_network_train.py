import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from eval import map_response
import concurrent.futures
from classifier_utils import *
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
import argparse
from functools import partial
import json
import os
import pickle
import xgboost as xgb
from sklearn.svm import SVC, LinearSVC
from debate import debate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from time import time
#from interpret.glassbox import ExplainableBoostingClassifier
# Load your dataset
def get_data(scan, rooms, specific_rooms, malicious):
    #df = pd.read_csv(f'answers_fixed_exp_balanced/glip/{scan}/answers_0_with_obs_full_mem_0.csv')
    df_0 = pd.read_csv(f'questions/glip/{scan}/no_queries.csv')
    df_1 = pd.read_csv(f'questions/glip/{scan}/yes_queries.csv')

    df = pd.concat([df_0, df_1])
   
    for room in rooms:
        specific_df = pd.read_csv(f'answers_fixed_exp_balanced/glip/{scan}/independent_answers/{room}.csv')
        specific_df = pd.read_csv(f'answers_fixed_exp_balanced/glip/{scan}/independent_answers/{room}.csv')
        df[f'{str(room)}_response'] = specific_df['Response']


    for col in df.columns:
        
        if col not in ['Room', 'Object', 'Answer']:
            df[col] = df[col].apply(map_response)

    if len(specific_rooms) > 0:
        df = df[df['Room'].isin(specific_rooms)]
    # Split features and labels
    categorical_columns = ['Room', 'Object']
    # One-hot encode categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        if col == 'Object':
            label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

            print(label_mapping)
    

    # agents = [ "['dining room', 'kitchen']_response"]
    # # agents = [ "['dining room']_response"]
    # # agents = [ "['kitchen']_response"]
    # # agents = [ "['dining room']_response", "['kitchen']_response"]
    # # agents = ["['bathroom', 'bedroom']_response"]
    # agents = ["['bathroom']_response"]
    # agents = ["['bedroom']_response"]
    # agents = ["['bathroom']_response", "['bedroom']_response"]

    # agents = ["['bathroom']_response", "['kitchen']_response"]


    # X = df[['Room', 'Object'] + agents].values
    # X = df[['Room', 'Object'] +  agents].values

    # X = df[['Room', 'Object'] +  agents].values

    # X = df[['Room', 'Object'] +  agents].values

    X = df.drop(columns=['Answer'])
    y = df['Answer'].values
    

    if malicious:
        X['54_response'] = 1 - X['54_response']

    return X, y, label_encoders

def train_val(trial, X, y, label_encoders, classifier_method = 'NN', scan = 'EU6Fwq7SyZv', rooms = ['kitchen', 'dining room'], test_percentage = 0.05, malicious = 0):
    # scaler = StandardScaler()
    # X_normalized = scaler.fit_transform(X)
    print(len(X))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_percentage, random_state=trial)

    if classifier_method == 'NN':
        #print(classifier_method)
        preds = train_val_nn(X_train, y_train, X_val, y_val)
        model = None
    elif classifier_method == 'MV':
        #print(X_val)
        votes = X_val[[r for r in X_val.columns if 'response' in r]]
        row_sums = votes.sum(axis=1)
        num_columns = votes.shape[1]
        preds = (row_sums > num_columns / 2).astype(int)
        preds = list(preds.values)
        model = None
    elif classifier_method == 'debate':
        # debate(X_val, scan, rooms)
        preds = debate(X_val, scan, rooms, trial, label_encoders, malicious)
        model = None
    elif classifier_method == 'XG':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,  # maximum depth of a tree
            'eta': 0.3,      # learning rate
            'eval_metric': 'logloss'  # evaluation metric
        }
        num_rounds = 100  # number of boosting rounds
        start = time()
        model = xgb.train(params, dtrain, num_rounds)
        probs = model.predict(dtest)
        preds = [1 if x > 0.5 else 0 for x in probs]
        print(time() - start)
    else:
        if classifier_method == 'LR':
            clf = LogisticRegression(random_state=trial)
        elif classifier_method == 'DT':
            clf = DecisionTreeClassifier(random_state=trial)
        elif classifier_method == 'EBM':
            ebm = ExplainableBoostingClassifier()
        elif classifier_method == 'SVM':
            clf = SVC(gamma='auto')
        elif classifier_method == 'LinearSVM':
            clf = LinearSVC()
        else:
            clf = RandomForestClassifier(n_estimators=1000, random_state=trial)

        model = clf.fit(X_train, y_train)

        preds = model.predict(X_val)

        #print(X_val)
    # for column in X_val.columns:
    #     if column.endswith('_response'):
    #         answers_of_agent = X_val[column]
    #         np.mean(np.array(answers_of_agent) == np.array(preds))

    tpr, fpr, tnr, fnr, acc = get_metrics(y_val, preds)

    return {'TPR': tpr, 'FPR': fpr, 'TNR': tnr, 'FNR': fnr, 'ACC': acc, 'val_preds': preds, 'model': model, 'X_train': X_train, 'y_train': y_train,'X_val': X_val, 'y_val': y_val, 'val_preds': preds}

# 
# def train_val_wrapper(trial):
#     print(trial)
#     return train_val(trial, X, y, classifier_method)
    
def main(X, y, label_encoders, classifier_method = 'NN', scan = 'EU6Fwq7SyZv', rooms = ['kitchen', 'dining room'], test_percentage = 0.05, malicious = 0):
    partial_function = partial(train_val, X=X, y=y, label_encoders = label_encoders, classifier_method = classifier_method, scan = scan, rooms = rooms, test_percentage = test_percentage, malicious=malicious)
    # print('in main')
    with concurrent.futures.ProcessPoolExecutor() as executor:

        return {trial: acc for trial, acc in enumerate(executor.map(partial_function, list(range(5))))}
        
    # train_val(trial, X, y, classifier_method)



# # Standardize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split data into training and validation sets
# accs = []


# def main():
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         return [acc for acc in executor.map(train_val_nn, list(range(5)))]
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-LLM answers')
    
    parser.add_argument('--agent_id', '-a', type=int, default= 0, metavar='N',
                    help='agent id') 
    
    parser.add_argument('--SCAN_IDs', '-scan', nargs='*', type=str, default= ['zsNo4HB9uLZ', '2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'oLBMNvg9in8', 'QUCTc6BB5sX', 'TbHJrupSAjP', 'X7HyMhZNoso', 'x8F5xyUWy9e', 'Z6MFQCViBuw'], metavar='N',
                    help='scan_ID') 
    
    parser.add_argument('--answer_behavior', type=str, default= 'helpful', metavar='N',
                    help='') 
    
    parser.add_argument('--method', type=str, default= 'glip', metavar='N',
                    help='') 
    
    parser.add_argument('--full_mem','-fm', type=int, default= 0, metavar='N',
                    help='') 
    
    parser.add_argument('--classifier_methods','-cm', nargs='*', type=str, default= ['RF', 'DT', 'LR', 'XG', 'SVM', 'LinearSVM', 'MV', 'NN', 'debate'], metavar='N',
                    help='') 
    
    parser.add_argument('--rooms','-r', nargs='*', type=str, default= ['dining room'], metavar='N',
                    help='') 
    parser.add_argument('--use_random','-rand',  type=int, default= 0, metavar='N',
                    help='') 
    parser.add_argument('--specific_query_room', '-qrs', nargs='*', type=str, default= [], metavar='N',
                    help='') 

    parser.add_argument('--test_percentage', '-test_p',type=float, default= 0.1, metavar='N',
                    help='') 

    parser.add_argument('--malicious_agent', '-m',type=float, default= 0, metavar='N',
                    help='') 

    args = parser.parse_args() 

    #2azQ1b91cZZ 
    print(args.classifier_methods)
    args.rooms.sort()
    print(args.rooms)
    for scan in args.SCAN_IDs:
        print(scan)
        #X,y, label_encoders = get_data(args.SCAN_ID, args.rooms, args.specific_query_room)
        X, y, label_encoders = get_data(scan, args.rooms, args.specific_query_room, args.malicious_agent)
        # print(X)

        # if args.use_random:
        #     num_rows = len(X)

        #     # Create an array with an equal number of 0s and 1s
        #     num_zeros_ones = num_rows // 2
        #     random_response = np.array([0] * num_zeros_ones + [1] * num_zeros_ones)

        #     # If the number of rows is odd, add an extra 0 or 1 randomly
        #     if num_rows % 2 != 0:
        #         random_response = np.append(random_response, np.random.choice([0, 1]))

        #     # Shuffle the array to randomize the order
        #     np.random.shuffle(random_response)

        #     # Assign the shuffled array to the new column
        #     X['random_response'] = random_response
        #     args.rooms += ['random']

        for cm in args.classifier_methods:
            print(cm)
            results = main(X, y, label_encoders, cm, scan, args.rooms, test_percentage = args.test_percentage, malicious=args.malicious_agent)
            results['Classifier Method'] = cm

            # print(results[0]['X_val'])

            mean_dict = {val: np.mean([results[trial][val] for trial in range(5)]) for val in ['ACC', 'TPR', 'FPR', 'TNR', 'FNR']}
            std_dict =  {val: np.std([results[trial][val] for trial in range(5)]) for val in ['ACC', 'TPR', 'FPR', 'TNR', 'FNR']}

            acc = [results[trial]['ACC'] for trial in range(5)]
            print(acc, np.mean(acc), np.std(acc))
            
            results['Mean'] = mean_dict
            results['Std Dev'] = std_dict

            save_folder = f'classification/glip/{scan}/{cm}/{args.test_percentage * 100}_percent_test/'
            os.makedirs(save_folder, exist_ok=True)

            if len(args.specific_query_room) > 0:
                save_file = os.path.join(save_folder, f"{'_'.join(args.rooms)}_{str(args.specific_query_room)}.pkl")
            if args.malicious_agent:
                save_file = os.path.join(save_folder, f"{'_'.join(args.rooms)}_malicious_agent.pkl")
            else:
                save_file = os.path.join(save_folder, f"{'_'.join(args.rooms)}.pkl")


            with open(save_file, 'wb') as f:
                pickle.dump(results, f)
