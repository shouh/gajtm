import sys
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import math
from tqdm import tqdm

class DataManager:
    def __init__(self, data_type):
        self.data_type = data_type
        if self.data_type == 'sq':
            self.train_data_path = '../../dataset/sq_relations/train_data.txt'
            self.val_data_path = '../../dataset/sq_relations/valid_data.txt'
            self.test_data_path = '../../dataset/sq_relations/test_data.txt'
            self.word_embedding_path = '../embedding/sq_wd_emb_300d.txt'
            self.rel_embedding_path = '../embedding/sq_kg_emb_300d.txt'
            self.relations_map = self.load_relations_map('../../dataset/sq_relations/relation.2M.list')
        else:
            self.train_data_path = '../../dataset/webqsp_relations/train_data.txt'
            self.test_data_path = '../../dataset/webqsp_relations/test_data.txt'
            self.word_embedding_path = '../embedding/wq_wd_emb_300d.txt'
            self.rel_embedding_path = '../embedding/wq_kg_emb_300d.txt'
            self.relations_map = self.load_relations_map('../../dataset/webqsp_relations/relations.txt')
        self.emb_dim = 300
        self.word_dic = {}
        self.word_embedding = []
        self.rel_dic = {}
        self.rel_embedding = []

        self.index2tag = {}
        label_list = ['<unk>', '<pad>', '0', '1']
        for i in range(len(label_list)):
            self.index2tag[label_list[i]]=i

        print('Filter out questions without negative training samples.')
        train_data, self.train_data_len = self.gen_train_data(self.train_data_path)
        print(f'Train data length:{len(train_data)}')
        test_data, self.test_data_len = self.gen_train_data(self.test_data_path)
        print(f'Test data length:{len(test_data)}')
        if self.data_type == 'sq':
            val_data, self.val_data_len = self.gen_train_data(self.val_data_path)
            print(f'Validation data length:{len(val_data)}')

        if not os.path.isfile(self.word_embedding_path):
            print(self.word_embedding_path, 'not exist!')
            if self.data_type == 'sq':
                self.save_embeddings(train_data + val_data + test_data)
            else:
                self.save_embeddings(train_data + test_data)
            print()
        self.word_dic, self.word_embedding = self.load_embeddings(self.word_embedding_path)
        self.rel_dic, self.rel_embedding = self.load_embeddings(self.rel_embedding_path)
        token_train_data = self.tokenize_train_data(train_data)
        print(f'Tokened train data length:{len(token_train_data)}')
        token_test_data = self.tokenize_train_data(test_data)
        print(f'Tokened test data length:{len(token_test_data)}')
        if self.data_type == 'sq':
            token_val_data = self.tokenize_train_data(val_data)
            print(f'Tokened val data length:{len(token_val_data)}')

        if self.data_type == 'sq':
            self.q_seqlen, self.pos_r_seqlen, self.pos_w_seqlen, self.neg_r_seqlen, self.neg_w_seqlen, self.maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w = self.find_maxlength(
                token_train_data + token_val_data + token_test_data)
            self.maxlen_q = 20
        else:
            self.q_seqlen, self.pos_r_seqlen, self.pos_w_seqlen, self.neg_r_seqlen, self.neg_w_seqlen, self.maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w = self.find_maxlength(
                token_train_data + token_test_data)

        self.maxlen_r = max(maxlen_pos_r, maxlen_neg_r)
        self.maxlen_w = max(maxlen_pos_w, maxlen_neg_w)

        self.token_train_data = self.pad_train_data(token_train_data, self.maxlen_q, self.maxlen_r, self.maxlen_w,
                                                    self.maxlen_r, self.maxlen_w)
        if self.data_type == 'sq':
            self.token_val_data = self.pad_train_data(token_val_data, self.maxlen_q, self.maxlen_r, self.maxlen_w,
                                                      self.maxlen_r, self.maxlen_w)
        self.token_test_data = self.pad_train_data(token_test_data, self.maxlen_q, self.maxlen_r, self.maxlen_w,
                                                   self.maxlen_r, self.maxlen_w)

    def idx2word(self, id_sentence, id_type='word'):
        if id_type == 'relation':
            dic = {idx: word for word, idx in self.rel_dic.items()}
        else:
            dic = {idx: word for word, idx in self.word_dic.items()}
        word_sentence = []
        for idx in id_sentence:
            if idx == 0:
                continue
            word_sentence.append(dic[idx])
        if not word_sentence:
            word_sentence.append(dic[id_sentence[0]])
        return word_sentence

    def gen_train_data(self, path):
        ''' Return training_data_list [[(question, pos_rels, pos_words, neg_rels, neg_words) * neg_size] * q_size]
        '''
        data_list = []
        data_len_list = []
        # pos_gt_1_counter = 0
        total_instance_counter = 0
        print('Load', path)
        # start = time.time()
        count = 0
        with open(path, encoding = 'utf-8') as infile:
            for line in infile:
                count += 1
                q_list = []
                seqlen_list = []
                tokens = line.strip().split('||')
                if self.data_type == 'sq':
                    pos_relations = tokens[7]
                    neg_relations = tokens[8]
                else:
                    pos_relations = tokens[8]
                    neg_relations = tokens[9]
                # if self.data_type == 'sq':
                question = tokens[1].strip().split(' ')
                labels =tokens[2].strip().split(' ')
                # else:
                #     question = tokens[2].replace('$ARG1', '').replace('$ARG2', '').strip().split(' ')

                pos_id = pos_relations.split(' ')[0]
                total_instance_counter += 1
                pos_rel, pos_word = self.split_relation(pos_id)
                for neg_id in neg_relations.split(' '):
                    # skip blank relation (relation_id 1797 = '')
                    if self.data_type == 'sq':
                        if neg_id == 'noNegativeAnswer':
                            continue
                    else:
                        if neg_id == '1797':
                            continue
                    neg_rels, neg_words = self.split_relation(neg_id)
                    total_instance_counter += 1
                    q_list.append((question, pos_rel, pos_word, neg_rels, neg_words, labels))
                    seqlen_list.append((len(question), len(pos_rel), len(pos_word), len(neg_rels), len(neg_words), len(labels)))
                    # print(q_list)
                    # sys.exit()
                if len(q_list) > 0:
                    data_list.append(q_list)
                    data_len_list.append(seqlen_list)
        print('average instances per question', total_instance_counter / len(data_list))
        return data_list, data_len_list

    def words_of_rel(self, relation):
        """
        提取关系单词级别的信息  word-level
        """
        rel_word_list = []
        if self.data_type == 'sq':
            rel_word_list = relation.split('/')[-1].split('_')
        else:
            items = relation.split("..")
            for item in items:
                rel_word_list += item.split('.')[-1].split("_")
        return rel_word_list  # relation-level

    def type_of_rel(self, relation):
        """
        得到answer的type
        :param relation:
        :return:
        """
        rel_type_list = []
        if self.data_type == 'sq':
            rel_type_list = '_'.join([relation.split('/')[-3]]).split('_') + relation.split('/')[-2].split('_')
        else:
            items = relation.split("..")
            for item in items:
                rel_type_list += u'_'.join([item.split('.')[-3]]).split('_') + item.split('.')[-2].split('_')
        return rel_type_list  # answer type

    def phrases_of_rel(self, relation):
        """
        提取关系的片段信息 phrase-level
        :param relation:
        :return:
        """
        rel_phrases_list = []
        if self.data_type == 'sq':
            rel_phrases_list = relation.strip('/').split('/')[-1:]
        else:
            items = relation.split("..")
            for item in items:
                rel_phrases_list += item.split(".")[-1:]
        return rel_phrases_list

    def split_relation(self, relation_id):
        '''Return relation_token_list and relation_token_name_list
        '''
        rel_list = []
        word_list = []

        relation_names = self.relations_map[int(relation_id)]
        relation_list = self.words_of_rel(relation_names)  # relation-level
        relation_phrases = self.phrases_of_rel(relation_names)
        rel_type_list = self.type_of_rel(relation_names)
        for word in relation_list:
            if word not in word_list:
                word_list.append(word)
        for ph in relation_phrases:
            if ph not in word_list:
                word_list.append(ph)
        for type in rel_type_list:
            if type not in word_list:
                word_list.append(type)
        if self.data_type == 'sp':
            r_list = relation_names.split("/")[1:]
            rel = ".".join(r_list)
            rel_list.append(rel)
        else:
            items = relation_names.split("..")
            for item in items:
                rel_list.append(item)
        return rel_list, word_list

    def find_unique(self, data):
        words = set()
        rels = set()
        # start = time.time()
        for idx, q_data in enumerate(data, 1):
            print('\r# of questions', idx, end='')
            try:
                words |= set(q_data[0][0])
                for data_obj in q_data:
                    rels |= set(data_obj[1]) | set(data_obj[3])
                    words |= set(data_obj[2]) | set(data_obj[4])
            except:
                print(idx, q_data)
        print()
        if '' in words:
            words.remove('')
        print(f'There are {len(rels)} unique relations and {len(words)} unique words.')
        return rels, words

    def load_embeddings(self, path):
        '''
        加载embedding
        :param path:
        :return:
        '''
        vocab_dic = {}
        embedding = []
        print('Load embedding from', path)
        with open(path, encoding = 'utf-8') as infile:
            for line in infile:
                tokens = line.strip().split()
                vocab_dic[tokens[0]] = len(vocab_dic)
                embedding.append([float(x) for x in tokens[1:]])
        embedding = np.array(embedding)
        if embedding.shape[0] != len(vocab_dic):
            print(f'Load embedding error: embedding.shape[0]={embedding.shape[0]} vocab_size={len(vocab_dic)}')
            sys.exit()
        if embedding.shape[1] != self.emb_dim:
            print(f'Load embedding error: embedding.shape[1]={embedding.shape[1]} emb_dim={self.emb_dim}')
            sys.exit()
        return vocab_dic, embedding

    def save_embeddings(self, data):
        rel_set, word_set = self.find_unique(data)
        Word2Vec_embedding = {}
        Rel2Vec_embedding = {}
        input_w2v_path = '/home/wshouhui/glove/glove.6B.300d.txt'
        input_r2v_path = '../data/kgembedding/kg_embedding_300d.txt'
        word_list = ['PADDING', '<unk>']
        embedding_dic = {}
        embedding_dic['PADDING'] = np.array([0.0] * self.emb_dim)
        embedding_dic['<unk>'] = np.random.uniform(low=-0.5, high=0.5, size=(self.emb_dim,))
        with open(input_w2v_path) as fin:
            for line in tqdm(fin):
                if line.strip():
                    seg_res = line.split(" ")
                    seg_res = [word.strip() for word in seg_res if word.strip()]
                    key = seg_res[0]
                    value = [float(word) for word in seg_res[1:]]
                    Word2Vec_embedding[key] = value
        with open(self.word_embedding_path, 'w') as outfile:
            for word in word_list:
                outfile.write(word + ' ')
                outfile.write(' '.join(str(v) for v in embedding_dic[word]))
                outfile.write('\n')
            for idx, word in enumerate(word_set):
                if word in Word2Vec_embedding:
                    outfile.write(word + ' ')
                    outfile.write(' '.join(str(v) for v in Word2Vec_embedding[word]))
                    outfile.write('\n')
                else:
                    outfile.write(word + ' ')
                    outfile.write(' '.join(str(v) for v in np.random.uniform(-0.5, 0.5, self.emb_dim)))
                    outfile.write('\n')

        with open(input_r2v_path) as fin:
            for line in tqdm(fin):
                if line.strip():
                    seg_res = line.split(" ")
                    seg_res = [rel.strip() for rel in seg_res if rel.strip()]
                    key = seg_res[0]
                    value = [float(rel) for rel in seg_res[1:]]
                    Rel2Vec_embedding[key] = value
        with open(self.rel_embedding_path, 'w') as routfile:
            for idx, rel in enumerate(rel_set):
                if rel in Rel2Vec_embedding:
                    routfile.write(rel + ' ')
                    routfile.write(' '.join(str(v) for v in Rel2Vec_embedding[rel]))
                    routfile.write('\n')
                else:
                    routfile.write(rel + ' ')
                    routfile.write(' '.join(str(v) for v in np.random.uniform(-0.5, 0.5, self.emb_dim)))
                    routfile.write('\n')
        return 0

    def tokenize_train_data(self, data):
        token_data = []
        for idx, q_data in enumerate(data):
            token_q_data = []
            question = list(
                map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], q_data[0][0]))
            labels = list(map(lambda x: self.index2tag[x] if x in self.index2tag else self.index2tag['<unk>'], q_data[0][5]))
            for data_obj in q_data:
                pos_rels = list(map(lambda x: self.rel_dic[x], data_obj[1]))
                pos_words = list(
                    map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], data_obj[2]))
                neg_rels = list(map(lambda x: self.rel_dic[x], data_obj[3]))
                neg_words = list(
                    map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], data_obj[4]))
                token_q_data.append((question, pos_rels, pos_words, neg_rels, neg_words, labels))
            token_data.append(token_q_data)
        # print(f'Time elapsed:{time.time()-start:.2f}')
        return token_data

    def pad_train_data(self, data, maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w):
        ''' Input: training_data_list [[(question, pos_rels, pos_words, neg_rels, neg_words) * neg_size] * q_size]
        '''
        padded_data = []
        for q_data in data:
            padded_q_data = []
            for obj in q_data:
                q = self.pad_obj(maxlen_q, obj[0])
                p_rel = self.pad_obj(maxlen_pos_r, obj[1])
                p_word = self.pad_obj(maxlen_pos_w, obj[2])
                n_rel = self.pad_obj(maxlen_neg_r, obj[3])
                n_word = self.pad_obj(maxlen_neg_w, obj[4])
                labels = self.pad_objl(maxlen_q, obj[5])
                padded_q_data.append((
                    q, p_rel, p_word, n_rel, n_word, labels
                ))
            padded_data.append(padded_q_data)
        return padded_data

    def pad_obj(self, max_len, sentence):
        return sentence[:max_len] + [0] * (max_len - len(sentence))

    def pad_objl(self, max_len, sentence):
        return sentence[:max_len] + [1] * (max_len - len(sentence))

    def find_maxlen(self, data):
        maxlen = 0
        seq_len = []
        for q_data in data:
            for obj in q_data:
                seq_len.append(len(obj))
                if len(obj) > maxlen:
                    maxlen = len(obj)
        return seq_len, maxlen

    def find_maxlength(self, data):
        q_seqlen = []
        pos_r_seqlen = []
        pos_w_seqlen = []
        neg_r_seqlen = []
        neg_w_seqlen = []
        maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w = 0, 0, 0, 0, 0
        for q_data in data:
            if len(q_data[0][0]) > maxlen_q:
                maxlen_q = len(q_data[0][0])
            if len(q_data[0][1]) > maxlen_pos_r:
                maxlen_pos_r = len(q_data[0][1])
            if len(q_data[0][2]) > maxlen_pos_w:
                maxlen_pos_w = len(q_data[0][2])
            for obj in q_data:
                q_seqlen.append(len(obj[0]))
                pos_r_seqlen.append(len(obj[1]))
                pos_w_seqlen.append(len(obj[2]))
                neg_r_seqlen.append(len(obj[3]))
                neg_w_seqlen.append(len(obj[4]))
                if len(obj[3]) > maxlen_neg_r:
                    maxlen_neg_r = len(obj[3])
                if len(obj[4]) > maxlen_neg_w:
                    maxlen_neg_w = len(obj[4])
        return q_seqlen, pos_r_seqlen, pos_w_seqlen, neg_r_seqlen, neg_w_seqlen, maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w

    def load_relations_map(self, path):
        ''' Return self.relations_map = {idx:relation_names}
        '''
        relations_map = {}
        print('Load', path)
        with open(path) as infile:
            for idx, line in enumerate(infile, 1):
                relations_map[idx] = line.strip()
        return relations_map


def batchify(data, batch_size):
    ''' Input: training_data_list [[(question, pos_rels, pos_words, neg_rels, neg_words) * neg_size] * q_size]
        Return: [[(question, pos_rels, pos_words, neg_rels, neg_words)*neg_size] * batch_size] * nb_batch]
    '''
    nb_batch = math.ceil(len(data) / batch_size)
    batch_data = [data[idx * batch_size:(idx + 1) * batch_size] for idx in range(nb_batch)]
    print('nb_batch', len(batch_data), 'batch_size', len(batch_data[0]))
    return batch_data


if __name__ == '__main__':
    data = DataManager()
