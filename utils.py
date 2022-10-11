import torch
import argparse
import datetime
import os


def cal_acc(sorted_score_label):
    if sorted_score_label[0][1] == 1:
        return 1
    else:
        return 0


def cal_mrr(sorted_score_label):
    for i in range(len(sorted_score_label)):
        if sorted_score_label[i][1] == 1:
            return 1 / (i + 1)


def extract_error(sorted_score_label):
    error_case = []
    for score, label, idx in sorted_score_label:
        if label == 1:
            break
        else:
            error_case.append(idx)
    return error_case


def predict_result(sorted_score_label):
    all_case = []
    for score, label, idx in sorted_score_label:
        all_case.append(idx)
        break
    return all_case


def save_best_model(model, args):
    """
    Save the optimal relation detection model
    :param model:
    :param args:
    :return:
    """
    if not os.path.exists('save_models'):
        os.makedirs('save_models')

    args.save_model_path = f'save_models/{args.data_type}_model.pt'
    with open('log.txt', 'a') as outfile:
        outfile.write(str(args) + '\n')

    print('save model at {}'.format(args.save_model_path))
    with open(args.save_model_path, 'wb') as outfile:
        torch.save(model, outfile)


def get_relation(relation_list, data_type):
    if data_type == 'sp':
        relation = relation_list[-1]
    else:
        relation = "..".join(relation_list)
    return relation


def handle_result(dataset, result_list, name, corpus, data_type):
    """
    Handle the result of relation detection and save the result in ../../data/relationdect_result/
    :param dataset:
    :param result_list:
    :param name:
    :param corpus:
    :param args:
    :return:
    """
    test_data = [obj for batch in dataset for obj in batch]
    for idx, cases in enumerate(result_list):
        if len(cases) > 0:
            question = test_data[idx][0][0]
            question = corpus.idx2word(question)
            question = ' '.join(question)
            correct_relation = test_data[idx][0][1]
            correct_relation = corpus.idx2word(correct_relation, id_type='relation')
            correct_relation = get_relation(correct_relation, data_type)
            if name == 'error':
                for error_idx in cases:
                    error_relation = test_data[idx][error_idx - 1][3]
                    error_relation = corpus.idx2word(error_relation, id_type='relation')
                    error_relation = get_relation(error_relation, data_type)
                    with open(f'../../results/{data_type}_result/relation_detection_error.txt', 'a', encoding = 'utf-8') as outfile:
                        outfile.write(question + '\t' + correct_relation + '\t' + error_relation + '\n')
            else:
                for all_idx in cases:
                    if all_idx == 0:
                        right_relation = test_data[idx][0][1]
                    else:
                        right_relation = test_data[idx][all_idx - 1][3]
                    right_relation = corpus.idx2word(right_relation, id_type='relation')
                    right_relation = get_relation(right_relation, data_type)
                    with open(f'../../results/{data_type}_result/relation_detection_result.txt', 'a', encoding = 'utf-8') as outfile:
                        outfile.write(question + '\t' + correct_relation + '\t' + right_relation + "\t" + '\n')

def convert(fileName, outputFile):
    fin = open(fileName, encoding = 'utf-8')
    fout = open(outputFile, "w", encoding = 'utf-8')

    for line in fin.readlines():
        query_list = []
        query_text = []
        line = line.strip().split('\t')
        sent = line[0].strip().split()
        pred = line[1].strip().split()
        for token, label in zip(sent, pred):
            if label == '1':
                query_text.append(token)
            if label == '0':
                query_text = list(filter(lambda x: x != '<pad>', query_text))
                if len(query_text) != 0:
                    query_list.append(" ".join(list(filter(lambda x: x != '<pad>', query_text))))
                    query_text = []
        query_text = list(filter(lambda x: x != '<pad>', query_text))
        if len(query_text) != 0:
            query_list.append(" ".join(list(filter(lambda x: x != '<pad>', query_text))))
            query_text = []
        if len(query_list) == 0:
            query_list.append(" ".join(list(filter(lambda x: x != '<pad>', sent))))
        fout.write(" ".join(query_list) + "\n")
