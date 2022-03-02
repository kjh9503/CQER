#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models import KGReasoning
from src.dataloader import TrainDataset, TestDataset, SingledirectionalOneShotIterator
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from src.util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple

query_name_dict = {('e', ('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                   }

name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys()) 
best_recall = 0.0
stopping_steps = 0

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    #parser.add_argument('--test', type=int, default=0, help="test")

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=64, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=256, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=32, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=1, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")
    parser.add_argument('--K', default=20, type=int, help="evaluation metrics @K")
    parser.add_argument('--l2_lambda', default=0.0, type=float, help="L2 Regularization")
    
    #parser.add_argument('--save_checkpoint_steps', default=50000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--save_embedding_dir', nargs='?', default='data/pretrain/',help='Path of learned embeddings. No need to set manually, will configure automatically')
    parser.add_argument('--save_embeddings', default=0, type=int, help="save embeddings")
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nitems', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--geo', default='beta', type=str, choices=['vec', 'box', 'beta'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--print_on_screen', action='store_true')
    
    parser.add_argument('--tasks', default='1p.2p.3p', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    return parser.parse_args(args)

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

def load_model(args, model):
    logging.info('Loading checkpoint %s...' % args.save_path)
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, step
        
def save_embeddings(model, args, average_metrics):
    metric = 'RECALL'
    if model.best_recall < average_metrics[metric][0]:
        print('updated embeddings. RECALL: ',average_metrics[metric])
        model.best_recall = average_metrics[metric][0]
        data_name = args.data_path.split('/')[1]
        file_name = 'embed.npz'
        args.save_embedding_dir = 'data/pretrain/%s/' % (data_name)
        if not os.path.exists(args.save_embedding_dir):
            os.makedirs(args.save_embedding_dir)
        np.savez(args.save_embedding_dir+file_name, item_embed=model.entity_embedding[:args.nitems,:].detach().cpu().numpy(), user_embed=model.entity_embedding[args.nitems:,:].detach().cpu().numpy(), rel_embed=model.relation_embedding.detach().cpu().numpy())

def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    if mode == 'Training average':
        for metric in metrics:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
    else:
        for metric in metrics:
            logging.info('%s %s at step %d: [ %f, %f, %f, %f, %f ]' % (mode, metric, step, metrics[metric][0], metrics[metric][1], metrics[metric][2], metrics[metric][3], metrics[metric][4]))

def evaluate(model, answers, total_train_interactions, args, dataloader, mode, step):
    '''
    Evaluate queries in dataloader
    '''
    
    #average_metrics = defaultdict(float)
    average_metrics = defaultdict(list)
    #all_metrics = defaultdict(float)
    #all_metrics = defaultdict(list)

    metrics = model.test_step(model, answers, total_train_interactions, args, dataloader, mode)
    num_users = len(metrics)
    num_queries = 0
    for user in metrics:
        #log_metrics(mode+" "+str(user), step, metrics[user])
        for metric in metrics[user]:
            #all_metrics["_".join([metric])] = metrics[user][metric]
            #if metric != 'num_queries':
            #    average_metrics[metric] += metrics[user][metric]
            
            #average_metrics[metric] += metrics[user][metric]
            average_metrics[metric].append(metrics[user][metric])
        #num_queries += metrics[user]['num_queries']

    for metric in average_metrics:
        #average_metrics[metric] /= num_users   
        average_metrics[metric] = np.sum(average_metrics[metric], axis=0, dtype=np.float32) / num_users 
        #all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, step, average_metrics)
    
    if args.save_embeddings:
        save_embeddings(model, args, average_metrics)

    return average_metrics #all_metrics

def write_results(result_path, result_file, test_average_metrics):
    with open(os.path.join(result_path, result_file), 'w') as fw_results:
        fw_results.write(f'validation score: {best_recall}\n')
        hits_list = test_average_metrics['HITS']
        ndcg_list = test_average_metrics['nDCG']
        precision_list = test_average_metrics['PRECISION']
        recall_list = test_average_metrics['RECALL']
        k_list = [10, 5, 20, 50, 100]
        for k, hits in zip(k_list, hits_list):
            line = f'hits@{k}: {hits}\n'
            fw_results.write(line)
        for k, ndcg in zip(k_list, ndcg_list):
            line = f'ndcg@{k}: {ndcg}\n'
            fw_results.write(line)
        for k, precision in zip(k_list, precision_list):
            line = f'precision@{k}: {precision}\n'
            fw_results.write(line)
        for k, recall in zip(k_list, recall_list):
            line = f'recall@{k}: {recall}\n'
            fw_results.write(line)
    

def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    train_queries = pickle.load(open(os.path.join(args.data_path, "train_queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_path, "train_answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test_queries.pkl"), 'rb')) 
    total_train_interactions = pickle.load(open(os.path.join(args.data_path, "train_user_dict.pkl"), 'rb'))
    valid_answers = pickle.load(open(os.path.join(args.data_path, "valid_user_dict.pkl"), 'rb'))
    test_answers = pickle.load(open(os.path.join(args.data_path, "test_user_dict.pkl"), 'rb'))

    
    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]

            if query_structure in train_queries:
                del train_queries[query_structure]

    return train_queries, train_answers, test_queries, valid_answers, test_answers, total_train_interactions


def main_helper(args):
    global best_recall, stopping_steps
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    print ("overwritting args.save_path")
    args.save_path = os.path.join(prefix, args.data_path.split('/')[-1], args.tasks, args.geo)
    if args.geo in ['box']:
        tmp_str = "box/box-d-{}-g-{}-lr-{}-reg-{}".format(args.hidden_dim, args.gamma, args.learning_rate, args.l2_lambda)
    elif args.geo in ['vec']:
        tmp_str = "vec/vec-d-{}-g-{}-lr-{}-reg-{}".format(args.hidden_dim, args.gamma, args.learning_rate, args.l2_lambda)
    elif args.geo == 'beta':
        tmp_str = "d-{}-g-{}-mode-{}-lr-{}-reg-{}".format(args.hidden_dim, args.gamma, args.beta_mode, args.learning_rate, args.l2_lambda)
    
    args.save_path = os.path.join(args.save_path, tmp_str, cur_time)
    result_path = os.path.join('results', args.data_path.split('/')[-1])
    result_file = tmp_str+'.result'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    set_logger(args)

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
        nitems = int(entrel[2].split(' ')[-1])
    
    args.nentity = nentity
    args.nrelation = nrelation
    args.nitems = nitems

    logging.info('-------------------------------'*3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    train_queries, train_answers, test_queries, valid_answers, test_answers, total_train_interactions = load_data(args, tasks) 
    
    logging.info("Training info:")
    path_list = ['1p', '2p', '3p', '4p', '5p']
    train_path_queries = defaultdict(set)
    train_other_queries = defaultdict(set)
    if args.do_train:
        for query_structure in train_queries:
            if query_structure in query_name_dict:
                logging.info(query_name_dict[query_structure]+": "+str(len(train_queries[query_structure])))
            else:
                logging.info(str(len(query_structure))+"i: "+str(len(train_queries[query_structure])))
        for query_structure in train_queries:
            if (query_structure in query_name_dict) and (query_name_dict[query_structure] in path_list):
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
                                    TrainDataset(train_path_queries, nentity, args.negative_sample_size, train_answers),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.cpu_num,
                                    collate_fn=TrainDataset.collate_fn
                                ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                                        TrainDataset(train_other_queries, nentity, args.negative_sample_size, train_answers),
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.cpu_num,
                                        collate_fn=TrainDataset.collate_fn
                                    ))
        else:
            train_other_iterator = None
   
        
        """
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]

        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
                                    TrainDataset(train_path_queries, nentity, nitems, nrelation, args.negative_sample_size, train_answers),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.cpu_num,
                                    collate_fn=TrainDataset.collate_fn
                                ))
        
        train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                                    TrainOtherDataset(users_paths, train_answers, nentity, nitems, nrelation, args.negative_sample_size),
                                    batch_size=args.batch_size,
                                    num_workers=args.cpu_num,
                                    collate_fn=TrainOtherDataset.collate_fn,
                                    worker_init_fn = TrainOtherDataset.worker_init_fn
                                ))
        """
    
    test_queries = flatten_query(test_queries)
    valid_dataloader = DataLoader(
        TestDataset(
            test_queries,
            args.nentity, 
        ), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn,
    )

    test_dataloader = DataLoader(
        TestDataset(
            test_queries,
            args.nentity, 
        ), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn,
    )

    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        nitems=nitems,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo=args.geo,
        use_cuda = args.cuda,
        box_mode=eval_tuple(args.box_mode),
        beta_mode = eval_tuple(args.beta_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict = query_name_dict
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        model = model.cuda()
    
    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate
        )
        warm_up_steps = args.max_steps // 2

    logging.info('Ramdomly Initializing %s Model...' % args.geo)
    init_step = 0
    
    step = init_step 
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    
    if args.do_train:
        training_logs = []
        for step in range(init_step, args.max_steps):
            logs = {
                'positive_sample_loss': 0.0,
                'negative_sample_loss': 0.0,
                'loss': 0.0,
            }
            
            log = model.train_step(model, optimizer, train_path_iterator, args, step)
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step)
                log = model.train_step(model, optimizer, train_path_iterator, args, step)
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_average_metrics = evaluate(model, valid_answers, total_train_interactions, args, valid_dataloader, 'Valid', step)
                    if best_recall < valid_average_metrics['RECALL'][0]:
                        best_recall = valid_average_metrics['RECALL'][0]
                        save_variable_list = {
                            'step': step, 
                            #'current_learning_rate': current_learning_rate,
                            #'warm_up_steps': warm_up_steps
                        }
                        save_model(model, optimizer, save_variable_list, args)
                        stopping_steps = 0
                    else:
                        stopping_steps += 1
                    logging.info(f'Stopping steps at step {step}: {stopping_steps}')
                    logging.info(f'Best Recall at step {step}: {best_recall}')
                    # early stopping
                    if stopping_steps == 10:
                        logging.info(f'Early stopping triggered at step {step}. Best Recall: {best_recall}')
                        logging.info("Training finished!!")
                        break

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        model, step = load_model(args, model)
        test_average_metrics = evaluate(model, test_answers, total_train_interactions, args, test_dataloader, 'Test', step)
        write_results(result_path, result_file, test_average_metrics)
    logging.info("Testing finished!!")


def main():
    main_helper(parse_args())


if __name__ == '__main__':
    main()
