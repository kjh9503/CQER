import os
import numpy as np
import argparse
import pickle
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import networkx as nx

query_dict = {
    '1p' : ('e',('r',)),
    '2p' : ('e', ('r', 'r')),
    '3p' : ('e', ('r', 'r', 'r')),
    '4p' : ('e', ('r', 'r', 'r', 'r')),
    '5p' : ('e', ('r', 'r', 'r', 'r', 'r')),
}

def load_cf(filename):
    user = []
    item = []
    user_dict = defaultdict(list)
    lines = open(filename, 'r').readlines()
    for l in lines:
        tmp = l.strip()
        interaction = [int(i) for i in tmp.split()]
        user_id = interaction[0]
        item_id = interaction[1]
        label = interaction[2]
        #if is_train or label == 1:
        user_dict[user_id].append(item_id)
        user.append(user_id)
        item.append(item_id)
        
    user = np.array(user, dtype=np.int32)
    item = np.array(item, dtype=np.int32)
    return (user, item), user_dict
    
def split_query_and_answer(train_user_dict, split):
    
    print('split train interactions for 1) query and 2) answer')
    figure = ['answer']*5
    figure[split] = 'query'
    print('[{0}|{1}|{2}|{3}|{4}]'.format(*figure))
    
    train_query_dict = defaultdict(list)
    train_answer_dict = defaultdict(list)
    train_query_users = []
    train_query_items = []
    for user, items in train_user_dict.items():
        if len(items) < 5:
            if len(items) == 1:
                #continue
                #query_items = items
                query_items = []
                answer_items = items
            else :
                if split < len(items):
                    query_items = [items[split]]
                    answer_items = list(set(items)-set(query_items))
                else:
                    query_items = [items[-1]]
                    answer_items = items[:-1]
        else:       
            if split == 0:
                answer_split = int(len(items)*0.2)
                answer_items = items[answer_split:]
                query_items = items[:answer_split]
            elif split == 1:
                answer_split1 = int(len(items)*0.2)
                answer_split2 = int(len(items)*0.4)
                answer_items = items[:answer_split1] + items[answer_split2:]
                query_items = items[answer_split1:answer_split2]
            elif split == 2:
                answer_split1 = int(len(items)*0.4)
                answer_split2 = int(len(items)*0.6)
                answer_items = items[:answer_split1] + items[answer_split2:]
                query_items = items[answer_split1:answer_split2]
            elif split == 3:
                answer_split1 = int(len(items)*0.6)
                answer_split2 = int(len(items)*0.8)
                answer_items = items[:answer_split1] + items[answer_split2:]
                query_items = items[answer_split1:answer_split2]
            elif split == 4:
                split_ind = int(len(items)*0.8)
                answer_items = items[:split_ind]
                query_items = items[split_ind:]
        for item in query_items:
            train_query_users.append(user)
            train_query_items.append(item)
        train_query_dict[user] = query_items
        train_answer_dict[user] = answer_items 
    train_query_users = np.array(train_query_users, dtype=np.int32)
    train_query_items = np.array(train_query_items, dtype=np.int32)
    return (train_query_users, train_query_items), train_query_dict, train_answer_dict

def construct_data(kg_data, cf_train_data, train_user_dict, valid_user_dict, test_user_dict, train_answer_dict):
    
    print('constructing Collaborative KG ...')

    n_relations = max(kg_data['r']) + 1
    reverse_kg_data = kg_data.copy()
    reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
    reverse_kg_data['r'] += n_relations
    kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)

    kg_data['r'] += 2
    n_relations = max(kg_data['r']) + 1
    n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1

    cf_train_data = (np.array(list(map(lambda d: d + n_entities, cf_train_data[0]))).astype(np.int32), cf_train_data[1].astype(np.int32))

    train_user_dict = {k + n_items: np.unique(v).astype(np.int32) for k, v in train_user_dict.items()}
    valid_user_dict = {k + n_items: np.unique(v).astype(np.int32) for k, v in valid_user_dict.items()}
    test_user_dict = {k + n_items: np.unique(v).astype(np.int32) for k, v in test_user_dict.items()}
    train_answer_dict = {k + n_entities: np.unique(v).astype(np.int32) for k, v in train_answer_dict.items()}

    # add interactions to kg data
    cf2kg_train_data = pd.DataFrame(np.zeros((len(cf_train_data[0]), 3), dtype=np.int32), columns=['h', 'r', 't'])
    cf2kg_train_data['h'] = cf_train_data[0]
    cf2kg_train_data['t'] = cf_train_data[1]

    reverse_cf2kg_train_data = pd.DataFrame(np.ones((len(cf_train_data[0]), 3), dtype=np.int32), columns=['h', 'r', 't'])
    reverse_cf2kg_train_data['h'] = cf_train_data[1]
    reverse_cf2kg_train_data['t'] = cf_train_data[0]

    kg_train_data = pd.concat([kg_data, cf2kg_train_data, reverse_cf2kg_train_data], ignore_index=True)

    # construct kg dict
    train_kg_dict = defaultdict(list)
    train_relation_dict = defaultdict(list)
    for row in kg_train_data.iterrows():
        h, r, t = row[1]
        train_kg_dict[h].append((t, r))
        train_relation_dict[r].append((h, t))

    return train_user_dict, valid_user_dict, test_user_dict, train_answer_dict, train_relation_dict, n_entities

if __name__ == "__main__":
    np.random.seed(2022)

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default="BookCrossing", help="which dataset to preprocess")
    args = parser.parse_args()
    
    # read rating file
    data_dir = '../data/' + args.dataset
    train_file = os.path.join(data_dir, 'train.txt')
    valid_file = os.path.join(data_dir, 'valid.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    cf_train_data, train_user_dict = load_cf(train_file)
    cf_valid_data, valid_user_dict = load_cf(valid_file)
    cf_test_data, test_user_dict = load_cf(test_file)
    
    # number of users and items
    n_users = max(max(max(cf_train_data[0]), max(cf_valid_data[0])), max(cf_test_data[0])) + 1
    n_items = max(max(max(cf_train_data[1]), max(cf_valid_data[1])), max(cf_test_data[1])) + 1
    print('number of users:',n_users)
    print('number of items:',n_items)
    
    train_queries_list = []
    train_answers_list = []
    users_paths_list = []
    for split in range(5):
        
        # split train interactions into interactions for 1) query and its 2) answer
        cf_train_data, train_query_dict, train_answer_dict = split_query_and_answer(train_user_dict, split)

        # read KG file
        kg_file = os.path.join(data_dir, 'kg_final.txt')
        kg_data = pd.read_csv(kg_file, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()

        # construct Collaborative KG
        split_train_user_dict, split_valid_user_dict, split_test_user_dict, train_answer_dict, train_relation_dict, n_entities \
        = construct_data(kg_data, cf_train_data, train_user_dict, valid_user_dict, test_user_dict, train_answer_dict)

        g = nx.DiGraph()
        g.add_nodes_from(list(range(0,n_users + n_entities)))
        for r, nodes in train_relation_dict.items():
            g.add_edges_from(list(nodes), rel=r)

        train_queries = defaultdict(list)
        train_answers = defaultdict(list)
        users_paths = defaultdict(list)
        
        print('extracting 3p paths...')

        # 1p
        for user, items in train_answer_dict.items():
            if len(items) == 0:
                print('len(items) == 0!!')
            query = tuple([user,tuple([0])])
            train_queries[('e', ('r',))].append(query)
            train_answers[query] = items 

        # 3p
        for user in tqdm(train_answer_dict.keys()):
            paths = list(nx.all_simple_edge_paths(g, source=user, target=train_answer_dict[user], cutoff=3))
            for path in paths:
                if len(path) == 1:
                    print(path)
                    continue
                query_structure = query_dict[str(len(path))+'p']
                rel_list = []
                for i in range(len(path)):
                    rel_list.append(g[path[i][0]][path[i][1]]['rel'])
                query = tuple([user,tuple(rel_list)])
                train_queries[query_structure].append(query)
                train_answers[query].append(path[-1][1])

        for query in train_answers.keys():
            users_paths[query[0]].append(query[1])

        # Change lists to sets
        for key in train_queries.keys():
            train_queries[key] = set(train_queries[key])
        for key in train_answers.keys():
            train_answers[key] = set(train_answers[key])
        
        
        # get rid of entities id 
        new_train_answers = defaultdict(list)
        new_train_queries = defaultdict(list)
        new_users_paths = defaultdict(list)
        for query, answers in train_answers.items():
            user = int(query[0])-(n_entities-n_items)
            query = tuple([user, query[1]])
            new_train_answers[query] = answers
        for key, queries in train_queries.items():
            for query in queries:
                user = int(query[0])-(n_entities-n_items)
                query = tuple([user, query[1]])
                new_train_queries[key].append(query)
        for user, paths in users_paths.items():
            user = int(user)-(n_entities-n_items)
            new_users_paths[user] = paths
   
        train_queries_list.append(new_train_queries)
        train_answers_list.append(new_train_answers)
        users_paths_list.append(new_users_paths)

    # save files
    train_queries_file = os.path.join(data_dir, 'train-queries-list.pkl')
    train_answers_file = os.path.join(data_dir, 'train-answers-list.pkl')
    users_paths_file = os.path.join(data_dir, 'users-paths-list.pkl')
    train_file = os.path.join(data_dir, 'train_user_dict.pkl')
    valid_file = os.path.join(data_dir, 'valid_user_dict.pkl')
    test_file = os.path.join(data_dir, 'test_user_dict.pkl')
    with open(train_queries_file, 'wb') as f:
        pickle.dump(train_queries_list, f)
    with open(train_answers_file, 'wb') as f:
        pickle.dump(train_answers_list, f)
    with open(users_paths_file, 'wb') as f:
        pickle.dump(users_paths_list, f)
    with open(train_file, 'wb') as f:
        pickle.dump(split_train_user_dict, f)
    with open(valid_file, 'wb') as f:
        pickle.dump(split_valid_user_dict, f)
    with open(test_file, 'wb') as f:
        pickle.dump(split_test_user_dict, f)
    print('finished path extraction')

    
    
