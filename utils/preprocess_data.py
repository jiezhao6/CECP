# -*- coding: utf-8 -*-

import thulac
import json
import random
from tqdm import tqdm


# -----------------------------------------------------------------------------
def load_data(path, dataset='criminal', scale='small', mode='test'):
    x, y = [], []
    p = path + dataset + '_' + scale + '_' + mode
    with open(p, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = eval(line)
        x.append(line['fact'])
        y.append(line['charges'])
    
    return x, y

# -----------------------------------------------------------------------------
def load_elements(path, mode='elements'):
    p = path + mode
    
    with open(p, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    elements = []
    for line in lines:
        line = eval(line)
        elements.append(line)
    
    return elements

# -----------------------------------------------------------------------------
def preprocess_criminal(path_read, path_write, scale='small', mode='train'):
    print(path_read)
    f = open(path_write + 'criminal_' + scale + '_' + mode, mode='a+', encoding='utf-8')
    segmenter = thulac.thulac(seg_only=True)  
    
    with open(path_read, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        random.shuffle(lines)
        
    l = len(lines)
    for content_index in tqdm(range(l)):
        content = lines[content_index]
        content_list = content.strip().split('\t')
        if len(content_list) != 3:
            continue
        
        tmp_x = content_list[0].replace(' ', '')
        tmp_x = segmenter.cut(tmp_x, text=True)
        
        tmp_y = []
        for i in content_list[1].strip().split():
            tmp_y.append(int(i))
        
        to_write = {'fact':tmp_x, 'charges':tmp_y}
        f.write(str(to_write) + '\n')
        
    f.close()

# -----------------------------------------------------------------------------
def preprocess_cail(path_read, path_write, scale='small', mode='test'):
    print(path_read)
    f = open(path_write + 'cail_' + scale + '_' + mode, mode='a+', encoding='utf-8')
    segmenter = thulac.thulac(seg_only=True)  
    
    with open(path_read, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        random.shuffle(lines)
        
    l = len(lines)
    for content_index in tqdm(range(l)):
        content = lines[content_index]
        content = json.loads(content)
        
        tmp_x = content['fact'].strip()
        tmp_x = segmenter.cut(tmp_x, text=True)
        
        labels = content['meta']
        tmp_y = labels['accusation']
        tmp_relevant_articles = labels['relevant_articles']
        tmp_punish_of_money = labels['punish_of_money']
        tmp_criminals = labels['criminals']
        tmp_term_of_imprisonment = labels['term_of_imprisonment']
        
        to_write = {'fact':tmp_x, 'charges':tmp_y, 'relevant_articles':tmp_relevant_articles, \
                    'punish_of_money':tmp_punish_of_money, 'criminals':tmp_criminals,\
                    'term_of_imprisonment':tmp_term_of_imprisonment}
        f.write(str(to_write) + '\n')
    
    f.close()

# ----------------------------------------------------------------------------
def preprocess_element(path_read, path_write, mode='elements'):
    print(path_read)
    f = open(path_write + mode, mode='a+', encoding='utf-8')
    segmenter = thulac.thulac(seg_only=True)  
    
    with open(path_read, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()[0]
        content = json.loads(lines)
    
    charge_names = list(content.keys())[1:]
    n = len(charge_names)
    for i in tqdm(range(n)):
        charge = charge_names[i]
        
        elem = content[charge]
        elem_keys = list(elem.keys())
        
        elem_subject = segmenter.cut(elem[elem_keys[1]], text=True)
        elem_subjective = segmenter.cut(elem[elem_keys[2]], text=True)
        elem_object = segmenter.cut(elem[elem_keys[3]], text=True)
        elem_objective = segmenter.cut(elem[elem_keys[4]], text=True)
        
        to_write = {charge:{'subject':elem_subject, 'subjective':elem_subjective, 'object':elem_object, 'objective':elem_objective}}
        f.write(str(to_write) + '\n')
    
    f.close()
    
# -----------------------------------------------------------------------------
train = 'train'
valid = 'valid'
test = 'test'

path_criminal = '../data/criminal'
path_criminal_small_train = path_criminal + '/small/' + train
path_criminal_small_valid = path_criminal + '/small/' + valid
path_criminal_small_test = path_criminal + '/small/' + test
path_criminal_middle_train = path_criminal + '/middle/' + train
path_criminal_middle_valid = path_criminal + '/middle/' + valid
path_criminal_middle_test = path_criminal + '/middle/' + test
path_criminal_big_train = path_criminal + '/big/' + train
path_criminal_big_valid = path_criminal + '/big/' + valid
path_criminal_big_test = path_criminal + '/big/' + test

path_cail = '../data/cail2018'
path_cail_small_train = path_cail + '/small/data_' + train + '.json'
path_cail_small_valid = path_cail + '/small/data_' + valid + '.json'
path_cail_small_test = path_cail + '/small/data_' + test + '.json'


path_elements = '../data/elements/explains_elements.json'


path_save = '../data/preprocessed_data/'


# ----------------------------------------------------------------------------
if __name__=='__main__':
    # preprocess_criminal(path_criminal_small_train, path_save, scale='small', mode='train')
    # preprocess_criminal(path_criminal_small_valid, path_save, scale='small', mode='valid')
    # preprocess_criminal(path_criminal_small_test, path_save, scale='small', mode='test')
    # preprocess_criminal(path_criminal_middle_train, path_save, scale='middle', mode='train')
    # preprocess_criminal(path_criminal_middle_valid, path_save, scale='middle', mode='valid')
    # preprocess_criminal(path_criminal_middle_test, path_save, scale='middle', mode='test')
    # preprocess_criminal(path_criminal_big_train, path_save, scale='big', mode='train')
    # preprocess_criminal(path_criminal_big_valid, path_save, scale='big', mode='valid')
    # preprocess_criminal(path_criminal_big_test, path_save, scale='big', mode='test')
    
    # preprocess_cail(path_cail_small_train, path_save, scale='small', mode='train')
    # preprocess_cail(path_cail_small_valid, path_save, scale='small', mode='valid')
    # preprocess_cail(path_cail_small_test, path_save, scale='small', mode='test')
    
    # preprocess_element(path_elements, path_save, mode='elements')
    
    '''
    x, y = load_data(path_save, dataset='criminal', scale='small', mode='train')
    elements = load_elements(path_save)
    '''











