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

def _all_simple_paths_graph(G, source, targets, cutoff, path_num):
    found = 0
    visited = dict.fromkeys([source])
    stack = [iter(G[source])]
    targets = set(targets)
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                found += 1
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                found += 1
                yield list(visited) + [target]
            stack.pop()
            visited.popitem()
        if path_num != -1 and found > path_num:
            break
            
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
        user_dict[user_id].append(item_id)
        user.append(user_id)
        item.append(item_id)
        
    user = np.array(user, dtype=np.int32)
    item = np.array(item, dtype=np.int32)
    return (user, item), user_dict

def mine_paths(g, train_answer_dict, path_num):
    global n_users
    global n_entities
    
    train_queries = defaultdict(list)
    train_answers = defaultdict(list)
    users_paths = defaultdict(list)

    # 3p
    for user in tqdm(train_answer_dict.keys()):
        #for path in nx.all_simple_edge_paths(g, source=user, target=train_answer_dict[user], cutoff=3):
        for simp_path in _all_simple_paths_graph(g, source=user, targets=train_answer_dict[user], cutoff=3, path_num=path_num):
            path = list(zip(simp_path[:-1], simp_path[1:]))
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

    return new_train_queries, new_train_answers, new_users_paths
    
if __name__ == "__main__":
    np.random.seed(2022)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="new_BookCrossing", help="which dataset to preprocess")
    parser.add_argument("--path_num", type=int, default=-1, help="number of paths to mine. If -1, mine all possible paths")
    args = parser.parse_args()
    
    # read rating file
    data_dir = '../data/' + args.dataset
    path_num = args.path_num
    train_file = os.path.join(data_dir, 'train.txt')
    valid_file = os.path.join(data_dir, 'valid.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    cf_train_data, train_user_dict = load_cf(train_file)
    cf_valid_data, valid_user_dict = load_cf(valid_file)
    cf_test_data, test_user_dict = load_cf(test_file)
    
    # read KG file
    kg_file = os.path.join(data_dir, 'kg_final.txt')
    kg_data = pd.read_csv(kg_file, sep='\t', names=['h', 'r', 't'], engine='python')
    kg_data = kg_data.drop_duplicates()
    n_users = max(max(max(cf_train_data[0]), max(cf_valid_data[0])), max(cf_test_data[0])) + 1
    n_items = max(max(max(cf_train_data[1]), max(cf_valid_data[1])), max(cf_test_data[1])) + 1
    n_relations = max(kg_data['r']) + 1
    
    reverse_kg_data = kg_data.copy()
    reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
    reverse_kg_data['r'] += n_relations
    kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)
    kg_data['r'] += 2
    n_relations = max(kg_data['r']) + 1
    n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
    n_cf_train = len(cf_train_data[0])
    
    print('number of users:',n_users)
    print('number of items:',n_items)
    print('number of relations:',n_relations)
    print('number of entities:',n_entities)
    
    # construct Collaborative KG
    cf_train_data = (np.array(list(map(lambda d: d + n_entities, cf_train_data[0]))).astype(np.int32), cf_train_data[1].astype(np.int32))
    train_answer_dict = {k + n_entities: np.unique(v).astype(np.int32) for k, v in train_user_dict.items()}
    train_user_dict = {k + n_items: np.unique(v).astype(np.int32) for k, v in train_user_dict.items()}
    valid_user_dict = {k + n_items: np.unique(v).astype(np.int32) for k, v in valid_user_dict.items()}
    test_user_dict = {k + n_items: np.unique(v).astype(np.int32) for k, v in test_user_dict.items()}

    # add interactions to kg data
    cf2kg_train_data = pd.DataFrame(np.zeros((n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
    cf2kg_train_data['h'] = cf_train_data[0]
    cf2kg_train_data['t'] = cf_train_data[1]
    reverse_cf2kg_train_data = pd.DataFrame(np.ones((n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
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

    # construct kg dict
    train_kg_dict = defaultdict(list)
    train_relation_dict = defaultdict(list)
    for row in kg_train_data.iterrows():
        h, r, t = row[1]
        train_kg_dict[h].append((t, r))
        train_relation_dict[r].append((h, t))
    
    g = nx.DiGraph()
    g.add_nodes_from(list(range(0,n_users + n_entities)))
    for r, nodes in train_relation_dict.items():
        g.add_edges_from(list(nodes), rel=r)
   
    train_queries, train_answers, users_paths = mine_paths(g, train_answer_dict, path_num)
    train_queries_list = [train_queries]
    train_answers_list = [train_answers]
    users_paths_list = [users_paths]
    
    # save files
    stats_file = os.path.join(data_dir, 'stats.txt')
    train_queries_file = os.path.join(data_dir, 'train-queries-list.pkl')
    train_answers_file = os.path.join(data_dir, 'train-answers-list.pkl')
    users_paths_file = os.path.join(data_dir, 'users-paths-list.pkl')
    train_file = os.path.join(data_dir, 'train_user_dict.pkl')
    valid_file = os.path.join(data_dir, 'valid_user_dict.pkl')
    test_file = os.path.join(data_dir, 'test_user_dict.pkl')
    with open(stats_file, 'w') as f:
        f.write('numentity: {}\n'.format(n_users + n_items))
        f.write('numrelations: {}\n'.format(n_relations))
        f.write('numitems: {}'.format(n_items))
    with open(train_queries_file, 'wb') as f:
        pickle.dump(train_queries_list, f)
    with open(train_answers_file, 'wb') as f:
        pickle.dump(train_answers_list, f)
    with open(users_paths_file, 'wb') as f:
        pickle.dump(users_paths_list, f)
    with open(train_file, 'wb') as f:
        pickle.dump(train_user_dict, f)
    with open(valid_file, 'wb') as f:
        pickle.dump(valid_user_dict, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test_user_dict, f)
    print('finished path extraction')

    
    
