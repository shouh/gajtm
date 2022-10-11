import os
import time
import datetime
import argparse
import math
import torch
from sklearn.utils import shuffle
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from data_preprocess import DataManager, batchify
from model import PMTLM
from utils import cal_acc, extract_error, predict_result, save_best_model, handle_result, cal_mrr, convert
import warnings
from math import cos
from tqdm import tqdm
from evaluation import entity_eval
from adax import AdaX, AdaXW
import random

warnings.filterwarnings('ignore')

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

class MultiTaskLearing:
    def __init__(self, args, corpus):
        self.args = args
        self.corpus = corpus
        print('Build model')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.index2tag = np.array(['<unk>', '<pad>', '0', '1'])
        self.model = PMTLM(self.args.dropout_rate, self.args.learning_rate, self.args.margin,
                                self.args.hidden_size,
                                self.corpus.word_embedding,
                                self.corpus.rel_embedding,
                                self.corpus.rel_dic)
        self.model.to(self.device)
        print(self.model)
        self.test_data = self.corpus.token_test_data
        self.test_data = batchify(self.test_data, self.args.batch_question_size)
        if self.args.data_type == 'sq':
            self.train_data = self.corpus.token_train_data
            self.train_data_len = self.corpus.train_data_len
            self.flat_train_data = [obj for q_obj in self.train_data for obj in q_obj]
            self.train_data = batchify(self.flat_train_data, self.args.batch_obj_size)
            self.val_data = self.corpus.token_val_data
            self.val_data = batchify(self.val_data, self.args.batch_question_size)
        else:
            # shuffle training data
            self.corpus.token_train_data, self.corpus.train_data_len = shuffle(self.corpus.token_train_data,
                                                                               self.corpus.train_data_len,
                                                                               random_state=12345)
            # split training data to train and validation
            self.split_num = int(0.9 * len(self.corpus.token_train_data))
            self.train_data = self.corpus.token_train_data[:self.split_num]
            self.train_data_len = self.corpus.train_data_len[:self.split_num]
            self.flat_train_data = [obj for q_obj in self.train_data for obj in q_obj]
            self.train_data = batchify(self.flat_train_data, self.args.batch_obj_size)
            self.val_data = self.corpus.token_train_data[self.split_num:]
            self.val_data = batchify(self.val_data, self.args.batch_question_size)

    def _cosine_anneal_schedule(self, epoch, epochs_per_cycle):
        lr = math.pi * (epoch % epochs_per_cycle) / epochs_per_cycle
        lr = self.args.learning_rate / 2 * (cos(lr) + 1)
        return lr


    def train(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        best_val_acc = 0.0
        earlystop_counter = 0

        # optimizer = torch.optim.AdamW(parameters, lr=self.args.learning_rate, weight_decay=0)
        # 10, 0.6
        optimizer = AdaX(parameters, lr=self.args.learning_rate, weight_decay=0)

        #0.00007 10 0.6   # 0.0001 10  0.7
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10,gamma=0.7,last_epoch=-1,verbose=False)
        #
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.00001, last_epoch=-1)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-05)
        for index in range(self.args.epoch_num):
            total_loss, totalrloss, totaleloss = 0, 0, 0
            currentloss = 0
            self.model.train()
            print("Epoch:%d, The Learning Rate is:%f"%(index, lr_scheduler.get_last_lr()[0])) #

            for batch_count, batch_data in enumerate(self.train_data, 1):
                question, pos_rel, pos_words, neg_rel, neg_words, labels = zip(*batch_data)
                q = Variable(torch.LongTensor(question)).to(self.device)
                p_rel = Variable(torch.LongTensor(pos_rel)).to(self.device)
                p_words = Variable(torch.LongTensor(pos_words)).to(self.device)
                n_rel = Variable(torch.LongTensor(neg_rel)).to(self.device)
                n_words = Variable(torch.LongTensor(neg_words)).to(self.device)
                labels = Variable(torch.LongTensor(labels)).to(self.device)
                ones = Variable(torch.ones(len(question))).to(self.device)

                q_mask = []
                for word in q:
                    q_mask.append([1 if i != 0 else 0 for i in word])
                q_mask = Variable(torch.LongTensor(q_mask)).to(self.device)

                entity_scores, all_pos_score, _= self.model(q, p_words, p_rel)
                _, all_neg_score, _ = self.model(q, n_words, n_rel)
                optimizer.zero_grad()
                rloss = F.margin_ranking_loss(all_pos_score, all_neg_score, ones, margin=self.args.margin)
                # 得到entity detection预测结果
                pred = torch.argmax(entity_scores, dim=2)
                eloss = self.model.loss_function(entity_scores, labels, q_mask, self.device)

                loss = 0.4 * rloss + 0.6 * eloss
                loss.backward()

                optimizer.step()

                total_loss += loss.data.cpu().numpy()
                totalrloss += rloss.data.cpu().numpy()
                totaleloss += eloss.data.cpu().numpy()
                average_loss = total_loss / batch_count
                avg_rloss = totalrloss / batch_count
                avg_eloss = totaleloss / batch_count
                if batch_count % self.args.print_every == 0:
                    print_str = f'\r{now()} Training Epoch {index} batch {batch_count} Loss:{average_loss:.4f} RLoss:{avg_rloss:.4f} ELoss:{avg_eloss:.4f}'
                    print(print_str, end='')
            print(print_str, end='\n')
            lr_scheduler.step()
            F1, val_acc, _, _ = self.evaluation(self.model)
            res = val_acc + F1
            if res > best_val_acc:
                earlystop_counter = 0
                save_best_model(self.model, self.args)
                best_val_acc = res
            else:
                earlystop_counter += 1
            if earlystop_counter >= self.args.earlystop_tolerance:
                print('EarlyStopping!')
                break

    def evaluation(self, model, data='dev'):
        model = model.eval()
        start_time = time.time()
        total_loss, total_acc, rr, average_loss = 0.0, 0.0, 0.0, 0.0
        error_idx = []
        all_idx = []
        gold_list = []
        pred_list = []
        if data == 'dev':
            input_data = self.val_data
        else:
            input_data = self.test_data
            temp_file = './tmp_file.txt'
            results_file = open(temp_file, 'w', encoding = 'utf-8')
            ner_result = f"../../results/{self.args.data_type}_result/entity_detection_result.txt"
        nb_question = sum(len(batch_data) for batch_data in input_data)  # 得到所有问题的数量方便后面计算准确率
        with torch.no_grad():
            for batch_count, batch_data in tqdm(enumerate(input_data, 1), desc='Processing'):
                start, end = 0, 0
                data_objs = [obj for q_obj in batch_data for obj in q_obj]
                question, pos_rel, pos_words, neg_rel, neg_words, labels = zip(*data_objs)
                q = Variable(torch.LongTensor(question)).to(self.device)
                p_rel = Variable(torch.LongTensor(pos_rel)).to(self.device)
                p_words = Variable(torch.LongTensor(pos_words)).to(self.device)
                n_rel = Variable(torch.LongTensor(neg_rel)).to(self.device)
                n_words = Variable(torch.LongTensor(neg_words)).to(self.device)
                labels = Variable(torch.LongTensor(labels)).to(self.device)
                ones = Variable(torch.ones(len(question))).to(self.device)

                q_mask = []
                for word in q:
                    q_mask.append([1 if i != 0 else 0 for i in word])
                q_mask = Variable(torch.LongTensor(q_mask)).to(self.device)


                entity_scores, all_pos, _ = model(q, p_words, p_rel)
                _, all_neg, _ = model(q, n_words, n_rel)
                # rloss = F.margin_ranking_loss(all_pos, all_neg, ones, margin=self.args.margin)
                # 得到entity detection预测结果
                # pred = torch.argmax(entity_scores, dim=2)
                # eloss = self.model.loss_function(entity_scores, labels, q_mask, self.device)
                # total_loss += loss.data.cpu().numpy()
                # average_loss = total_loss / batch_count
                pred = torch.argmax(entity_scores[0,:,:], dim=1)
                # 将掩膜添加到预测结果上，便于计算准确率
                pred = pred.int().unsqueeze(0)
                q_mask = q_mask[0,:].unsqueeze(0)
                input_ids = q[0,:].unsqueeze(0)
                target = labels[0,:].unsqueeze(0)
                pred = pred * q_mask
                target = target * q_mask
                input_ids = input_ids * q_mask

                pred = pred[:, 0:q_mask.sum().int().item()]
                input_ids = input_ids[:, 0:q_mask.sum().int().item()]
                target = target[:, 0:q_mask.sum().int().item()]
                gold_list.append(target.cpu().numpy()[0,:])
                pred_list.append(pred.cpu().numpy()[0,:])

                for idx, q_obj in enumerate(batch_data):
                    end += len(q_obj)
                    score_list = [all_pos[start]]
                    label_list = [1]
                    batch_neg_score = all_neg[start:end]
                    for ns in batch_neg_score:
                        score_list.append(ns)
                    label_list += [0] * len(batch_neg_score)
                    start = end
                    score_label = [(x, y, idx) for idx, (x, y) in enumerate(zip(score_list, label_list))]
                    sorted_score_label = sorted(score_label, key=lambda x: x[0], reverse=True)
                    total_acc += cal_acc(sorted_score_label)
                    rr += cal_mrr(sorted_score_label)
                    if data == 'test':
                        # handel entity detection result
                        label = self.index2tag[pred.cpu().numpy()[0,:]]
                        question = self.corpus.idx2word(input_ids[0].cpu().numpy())
                        gold = self.index2tag[target.cpu().numpy()[0,:]]
                        results_file.write("{}\t{}\t{}\n".format(" ".join(question), " ".join(label), " ".join(gold)))

                        error_cases = extract_error(sorted_score_label)
                        all_cases = predict_result(sorted_score_label)
                        error_idx.append(error_cases)
                        all_idx.append(all_cases)
            time_elapsed = time.time() - start_time
            average_acc = total_acc / nb_question
            P, R, F1 = entity_eval(gold_list, pred_list, self.index2tag, type=False)
            mrr = rr / nb_question
            print('Val Batch {} Spend Time:{:.2f}s Acc:{:.4f} F1:{:.4f} MRR:{:.4f}# question:{}'.format(
                batch_count,
                time_elapsed,
                average_acc,
                F1,
                mrr,
                nb_question))
            if data == 'test':
                results_file.flush()
                results_file.close()
                convert(temp_file, ner_result)
                os.remove(temp_file)
            return F1, average_acc, error_idx, all_idx

    def predict_by_singlemodel(self):
        save_best_model = f'./save_models/{self.args.data_type}_model.pt'
        print('Load best model', save_best_model)
        with open(save_best_model, 'rb') as infile:
            model = torch.load(infile)
            print(model)
        F1, acc, error_list, all_list = self.evaluation(model, data='test')
        handle_result(self.test_data, error_list, "error", self.corpus, self.args.data_type)
        handle_result(self.test_data, all_list, "result", self.corpus, self.args.data_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # setting
    parser.add_argument('-train', default=False, action='store_true')  # True/False
    parser.add_argument('-test', default=False, action='store_true')  # True/False
    parser.add_argument('--data_type', type=str, default='sp')  # [sp/wq]
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--apply_dwa', type=str, default='dwa')
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.00005)  # WQ 0.0001
    parser.add_argument('--hidden_size', type=int, default=300)  # [100/150/200/300/400]
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--batch_question_size', type=int, default=1)
    parser.add_argument('--batch_obj_size', type=int, default=128)  # [64/128/256]
    parser.add_argument('--earlystop_tolerance', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=10)  # 100
    args = parser.parse_args()

    # Load data
    corpus = DataManager(args.data_type)

    t = MultiTaskLearing(args=args, corpus=corpus)

    # training
    if args.train:
        t.train()
    # test
    if args.test:
        t.predict_by_singlemodel()
