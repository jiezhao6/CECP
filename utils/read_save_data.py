# -*- coding: utf-8 -*-


import numpy as np
import pickle
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import string
import thulac
import json


global_punctuations = '[' + string.punctuation + '，。？！：；‘’“”（）《》、]'  
UNK = 'Z_UNK'


'''========================================================================================================================'''
def count_charges(scale):
    if scale=='small':
        use_data = ['train', 'valid']
    else:
        use_data = ['train']
    charges, articles = {}, {}
    too_shot_fact = []
    for i in use_data:
        if scale=='small':
            path = '../data/cail/data_{}.json'.format(i)
        else:
            path = '../data/cail/{}.json'.format(i)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            tmp_data = json.loads(line)
            charge = tmp_data['meta']['accusation']
            article = tmp_data['meta']['relevant_articles']
            criminal = tmp_data['meta']['criminals']
            fact = tmp_data['fact']
            if len(fact) < 30:
                too_shot_fact.append(fact)
                
            if '二审' not in fact and '原审' not in fact and len(fact) >= 30 and \
                sum([j==charge[0] for j in charge])==len(charge) and \
                sum([j==article[0] for j in article])==len(article) and \
                sum([j==criminal[0] for j in criminal])==len(criminal):
                
                if charges.__contains__(charge[0]):
                    charges[charge[0]] += 1
                else:
                    charges.update({charge[0]: 1})
                if articles.__contains__(article[0]):
                    articles[article[0]] += 1
                else:
                    articles.update({article[0]: 1})
    
        
    def filter_dict(data_dict):
        return {k: v for k, v in data_dict.items() if v >= 100}

    print(sum(charges.values()))
    print(sum(articles.values()))
    print()

    charges_filted= filter_dict(charges)
    articles_filted = filter_dict(articles)
    
    print(sum(charges_filted.values()))
    print(sum(articles_filted.values()))
    print('charge num: '+ str(len(charges_filted)))
    print('article num: '+ str(len(articles_filted)))
    print()
        
    if sum(charges_filted.values()) != sum(articles_filted.values()):
        charges_new = {}
        articles_new = {}
        
        for i in use_data:
            n = 0
            if scale=='small':
                path = '../data/cail/data_{}.json'.format(i)
            else:
                path = '../data/cail/{}.json'.format(i)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                tmp_data = json.loads(line)
                charge = tmp_data['meta']['accusation']
                article = tmp_data['meta']['relevant_articles']
                criminal = tmp_data['meta']['criminals']
                fact = tmp_data['fact']
                
                if '二审' not in fact and '原审' not in fact and len(fact) >= 30 and \
                    sum([j==charge[0] for j in charge])==len(charge) and \
                    sum([j==article[0] for j in article])==len(article) and \
                    sum([j==criminal[0] for j in criminal])==len(criminal):
                    
                    if articles_filted.__contains__(article[0]) and charges_filted.__contains__(charge[0]):
                        n += 1
                        if articles_new.__contains__(article[0]):
                            articles_new[article[0]] += 1
                        else:
                            articles_new.update({article[0]: 1})
            
                        if charges_new.__contains__(charge[0]):
                            charges_new[charge[0]] += 1
                        else:
                            charges_new.update({charge[0]: 1})

                    
            print('dataset nums: ', n)
    
    print()
    print(sum(charges_new.values()))
    print(sum(articles_new.values()))
    print('charge num: '+ str(len(charges_new)))
    print('article num: '+ str(len(articles_new)))

    return charges_new, articles_new

'''========================================================================================================================'''
def load_word_embedding_data(path_word_embedding_data):
    with open(path_word_embedding_data, 'rb') as f:
        data = pickle.load(f)
        
    word_embedding = data['word_embedding']
    word2id = data['word2id']
    id2word = data['id2word']
    assert word_embedding.shape[0]==len(word2id)
    return word_embedding, word2id, id2word


'''========================================================================================================================'''
def delete_punctuations(s, len_sentence):
    s = re.sub(pattern=global_punctuations, repl='', string=s)
    s = s.strip().split(' ')
    s = [i for i in s if i!='']
    s = s[:len_sentence]
    return s


'''========================================================================================================================'''
def make_data(word_embedding, word2id, dataset, scale, mode, charges, articles):
    path_preprocessed_data = '../data/preprocessed_data/'
    path_data = '../data/data/'
    path = path_preprocessed_data + dataset + '_' + scale + '_' + mode
    num_sentences= 64
    len_sentence= 32
    punctuations='[。，？！；.,?!;]'
    
    if True:
        x, y, law = [], [], []
        sent_num, sent_len = [], []   
            
        with open(path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = eval(line)
            tmp_x = np.zeros(shape=(num_sentences, len_sentence), dtype=np.int32)
                
            fact = line['fact'].strip()
            fact_for_filter = fact.replace(' ', '')
            fact = re.split(punctuations, fact)[:num_sentences]
            
            charge = line['charges']
            article = line['relevant_articles']
            criminal = line['criminals']
            
            sent_len_ = []
            if '二审' not in fact_for_filter and '原审' not in fact_for_filter and len(fact_for_filter) >= 30 and \
                    sum([j==charge[0] for j in charge])==len(charge) and \
                    sum([j==article[0] for j in article])==len(article) and \
                    sum([j==criminal[0] for j in criminal])==len(criminal):
                if articles.__contains__(article[0]) and charges.__contains__(charge[0]):
                    for i in range(len(fact)):
                        f = delete_punctuations(fact[i], len_sentence)
                        sent_len_.append(len(f))
                        for j in range(len(f)):
                            word = f[j]
                            if word in word2id:
                                tmp_x[i, j] = word2id[word]
                            else:
                                tmp_x[i, j] = word2id[UNK]
                    x.append(tmp_x)
                    y.append(charge[0])
                    sent_len.append(sent_len_)
                    sent_num.append(len(fact))
                    law.append(article[0])

    return np.array(x), np.array(y), sent_num, sent_len, law



'''========================================================================================================================'''
def encode_label(charges, articles):
    charges_class = list(charges.keys())
    articles_class = list(articles.keys())
    
    charge2num, num2charge, article2num, num2article = {}, {}, {}, {}
    for i, item in enumerate(charges_class):
        charge2num.update({item : i})
        num2charge.update({i : item})
    for i, item in enumerate(articles_class):
        article2num.update({item : i})
        num2article.update({i : item})
    return charge2num, num2charge, article2num, num2article


'''========================================================================================================================'''
def make_elements(word_embedding, word2id, y, law, charge2num, num2charge, article2num):
    def look_up_id(l, length):
        res = [0] * length
        for i in range(len(l)):
            word = l[i]
            if word in word2id:
                res[i] = word2id[word]
            else:
                res[i] = word2id[UNK]
        return res
    
    path_preprocessed_data = '../data/preprocessed_data/'
    path_data = '../data/data/'
    len_subj = 100
    len_subtive = 100
    len_obj = 200
    len_objtive = 400
    path_elem = path_preprocessed_data + 'elements'

    if True:
        with open(path_elem, mode='r', encoding='utf-8') as f:
            elements = f.readlines()
        dict_elements = {}
        for line in elements:
            line = eval(line)
            line = [[k, v] for k, v in line.items()][0]
            dict_elements.update({line[0].replace('、', ''):line[1]})

        with open('../data/preprocessed_data/supplementary_explains_elements.txt', mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        supplementary_charge_elements = {}
        segmenter = thulac.thulac(seg_only=True)
        for line in lines:
            line = line.strip().split('&')
            assert len(line)==5
            c = {'subject':segmenter.cut(line[1], text=True),
                 'subjective':segmenter.cut(line[2], text=True),
                 'object':segmenter.cut(line[3], text=True),
                 'objective':segmenter.cut(line[4], text=True)}
            supplementary_charge_elements.update({line[0] : c})

        dict_elements.update(supplementary_charge_elements)

        y_encoded = []
        for i in y:
            y_encoded.append(charge2num[i])
        law_encoded = []
        for i in law:
            law_encoded.append(article2num[i])
        
        ele_subject, ele_subjective, ele_object, ele_objective = [], [], [], []
        for i in range(len(num2charge)):
            k = num2charge[i]
            k = k.replace('[', '').replace(']', '').replace('、', '') + '罪'
            e = dict_elements[k]

            subj, subtive, obj, objtive = e['subject'], e['subjective'], e['object'], e['objective']

            subj    = delete_punctuations(subj, len_subj)
            subtive = delete_punctuations(subtive, len_subtive)
            obj     = delete_punctuations(obj, len_obj)
            objtive = delete_punctuations(objtive, len_objtive)
            subj    = look_up_id(subj, len_subj)
            subtive = look_up_id(subtive, len_subtive)
            obj     = look_up_id(obj, len_obj)
            objtive = look_up_id(objtive, len_objtive)
            
            ele_subject.append(subj)
            ele_subjective.append(subtive)
            ele_object.append(obj)
            ele_objective.append(objtive)
        
        return np.array(ele_subject), np.array(ele_subjective), np.array(ele_object), np.array(ele_objective), y_encoded, law_encoded

'''========================================================================================================================'''
def save_fact_data(path, x, y, sent_num, sent_len, law, num2charge, num2article):
    to_save = {'x':np.array(x),
               'y':np.array(y),
               'sent_num':sent_num,
               'sent_len':sent_len,
               'law':law,
               'num2charge':num2charge,
               'num2article':num2article}
    with open(path, mode='wb') as f:
        pickle.dump(to_save, f, protocol=4)
    
def save_elements_data(path, ele_subject, ele_subjective, ele_object, ele_objective):
    to_save = {'ele_subject':ele_subject,
               'ele_subjective':ele_subjective,
               'ele_object':ele_object,
               'ele_objective':ele_objective}
    with open(path, mode='wb') as f:
        pickle.dump(to_save, f)



'''========================================================================================================================='''
def make_data_compare(word_embedding, word2id, scale, mode, charges, articles, charge2num, article2num, max_length=512):
    path_preprocessed_data = '../data/preprocessed_data/'
    path_data = '../data/data/'
    path = path_preprocessed_data + 'cail' + '_' + scale + '_' + mode
    
    if True:
        x, y, law, x_len = [], [], [], []
            
        with open(path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = eval(line)
            tmp_x = np.zeros(shape=(max_length, ), dtype=np.int32)
                
            fact = line['fact'].strip()
            fact_for_filter = fact.replace(' ', '')
            fact = delete_punctuations(fact, max_length)
            
            charge = line['charges']
            article = line['relevant_articles']
            criminal = line['criminals']
            
            sent_len_ = []
            if '二审' not in fact_for_filter and '原审' not in fact_for_filter and len(fact_for_filter) >= 30 and \
                    sum([j==charge[0] for j in charge])==len(charge) and \
                    sum([j==article[0] for j in article])==len(article) and \
                    sum([j==criminal[0] for j in criminal])==len(criminal):
                if articles.__contains__(article[0]) and charges.__contains__(charge[0]):
                    for j in range(len(fact)):
                        word = fact[j]
                        if word in word2id:
                            tmp_x[j] = word2id[word]
                        else:
                            tmp_x[j] = word2id[UNK]
                    x.append(tmp_x)
                    y.append(charge[0])
                    x_len.append(len(fact))
                    law.append(article[0])
    y_encoded = []
    for i in y:
        y_encoded.append(charge2num[i])
    law_encoded = []
    for i in law:
        law_encoded.append(article2num[i])
    return np.array(x), np.array(y_encoded), x_len, law_encoded

def save_fact_data_for_compare(path, x, y, x_len, law, num2charge, num2article):
    to_save = {'x':x,
               'y':y,
               'x_len':x_len,
               'law':law,
               'num2charge':num2charge,
               'num2article':num2article}
    with open(path, mode='wb') as f:
        pickle.dump(to_save, f, protocol=4)  


# -----------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    path_word_embedding_data = '../data/data/word_embedding_data.pkl'
    word_embedding, word2id, id2word = load_word_embedding_data(path_word_embedding_data)
    
    charges_small, articles_small = count_charges('small')
    
    charge2num_small, num2charge_small, article2num_small, num2article_small = encode_label(charges_small, articles_small)
    
    
    # ====================================================================================
    dataset, scale, mode = 'cail', 'small', 'train'
    x, y, sent_num, sent_len, law = make_data(word_embedding, word2id, dataset, scale, mode, charges_small, articles_small)
    subj, subjtive, obj, objtive, y, law = make_elements(word_embedding,word2id,y,law,charge2num_small,num2charge_small,article2num_small)
    path_save_fact = '../data/data/' + dataset + '_' + scale + '_' + mode + '.pkl'
    path_save_element_cail_small = '../data/data/' + 'elements_cail_small' + '.pkl'
    save_fact_data(path_save_fact, x, y, sent_num, sent_len, law, num2charge_small, num2article_small)
    save_elements_data(path_save_element_cail_small, subj, subjtive, obj, objtive)
    
    dataset, scale, mode = 'cail', 'small', 'valid'
    x, y, sent_num, sent_len, law = make_data(word_embedding, word2id, dataset, scale, mode, charges_small, articles_small)
    subj, subjtive, obj, objtive, y, law = make_elements(word_embedding,word2id,y,law,charge2num_small,num2charge_small,article2num_small)
    path_save_fact = '../data/data/' + dataset + '_' + scale + '_' + mode + '.pkl'
    save_fact_data(path_save_fact, x, y, sent_num, sent_len, law, num2charge_small, num2article_small)
    
    dataset, scale, mode = 'cail', 'small', 'test'
    x, y, sent_num, sent_len, law = make_data(word_embedding, word2id, dataset, scale, mode, charges_small, articles_small)
    subj, subjtive, obj, objtive, y, law = make_elements(word_embedding,word2id,y,law,charge2num_small,num2charge_small,article2num_small)
    path_save_fact = '../data/data/' + dataset + '_' + scale + '_' + mode + '.pkl'
    save_fact_data(path_save_fact, x, y, sent_num, sent_len, law, num2charge_small, num2article_small)
    
    
    
    
    '''=================================================================== For CNN'''
    dataset, scale, mode = 'cail', 'small', 'train'
    path_save_fact = '../data/cail/' + dataset + '_' + scale + '_' + mode + '.pkl'
    x, y, x_len, law = make_data_compare(word_embedding,word2id,scale,mode,charges_small,articles_small,charge2num_small,article2num_small)
    save_fact_data_for_compare(path_save_fact, x, y, x_len, law, num2charge_small, num2article_small)
    print(len(x))
    
    dataset, scale, mode = 'cail', 'small', 'valid'
    path_save_fact = '../data/cail/' + dataset + '_' + scale + '_' + mode + '.pkl'
    x, y, x_len, law = make_data_compare(word_embedding,word2id,scale,mode,charges_small,articles_small,charge2num_small,article2num_small)
    save_fact_data_for_compare(path_save_fact, x, y, x_len, law, num2charge_small, num2article_small)
    print(len(x))
    
    dataset, scale, mode = 'cail', 'small', 'test'
    path_save_fact = '../data/cail/' + dataset + '_' + scale + '_' + mode + '.pkl'
    x, y, x_len, law = make_data_compare(word_embedding,word2id,scale,mode,charges_small,articles_small,charge2num_small,article2num_small)
    save_fact_data_for_compare(path_save_fact, x, y, x_len, law, num2charge_small, num2article_small)
    print(len(x))
    
    
    
    
    
    









