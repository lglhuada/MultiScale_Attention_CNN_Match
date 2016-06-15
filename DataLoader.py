from __future__ import division
import numpy as np
from joblib import Parallel,delayed,load,dump
import subprocess
from multiprocessing import *

wordvec_path = 'wordvec/pool.dict.tst.f64'
wmodel = load(wordvec_path, mmap_mode = 'r+')
dictindex = wmodel[0]
dictarray = wmodel[1].astype(np.float32)

querytitle_path = 'querytitle-00000'
index_path = 'pair-00000'

default_word = '</s>'

wordvecdim = 100

class DataLoader():
    def __init__(self, querytitle_path, index_path, len_limit, batch_size):
        self.query, self.title = self.extract_query_title(querytitle_path)
        self.index = self.extract_index(index_path)
        self.train = self.index[:int(0.8*len(self.index))]
        self.test  = self.index[-1*int(0.2*len(self.index)):]
        self.len_limit = len_limit
        self.batch_size = batch_size
        self.batch_index_train = 0
        self.batch_index_test  = 0
        self.batch_num_train   = int(len(self.train)/self.batch_size) - 1
        self.batch_num_test    = int(len(self.test)/self.batch_size) - 1

    def extract_query_title(self,querytitle_path):
        print('Begin to extract query and title data')
        query,title = [],[]
        for line in open(querytitle_path):
                split_line = line.split('\t')
                if len(split_line) >= 3:
                    query.append(split_line[1])
                    for t in split_line[2:]: title.append(t)
        print('Finish extracting query and title data',len(query),len(title))
        return query,title

    def extract_index(self, index_path):
        print('Begin to extract index')
        index = []
        i = 0
        for line in open(index_path):
                index.append([int(m) for m in line.strip().split('\t')])
        print('Finish extracting index',len(index))
        return index

    def convert_to_vec(self, input_sen):
        sen_array = input_sen.split(' ')
        if len(sen_array) > self.len_limit: sen_array = sen_array[:self.len_limit]
        else: sen_array.extend([default_word for i in xrange(self.len_limit-len(sen_array))])
        default_vec = dictarray[dictindex[default_word]]
        senvec = np.zeros((wordvecdim, self.len_limit))
        for index_w, w in enumerate(sen_array):
            if dictindex.has_key(w)==True:
                senvec[:,index_w] = dictarray[int(dictindex[w])]
            else:
                senvec[:,index_w] = default_vec
        return senvec

    def next_batch(self,data_category):
        if data_category == 'train':
            indexdata = self.train
            batch_index = self.batch_index_train
            self.batch_index_train += self.batch_size
        if data_category == 'test':
            indexdata = self.test
            batch_index = self.batch_index_test
            self.batch_index_test += self.batch_size
        QueryVec, Title1Vec, Title2Vec = \
        np.zeros((self.batch_size,wordvecdim,self.len_limit,1),dtype=np.float32),\
        np.zeros((self.batch_size,wordvecdim,self.len_limit,1),dtype=np.float32),\
        np.zeros((self.batch_size,wordvecdim,self.len_limit,1),dtype=np.float32)

        batch_index_Tuple = indexdata[batch_index:batch_index+self.batch_size]
        index_flag = 0
        for batch_index_tuple_index, batch_index_tuple in enumerate(batch_index_Tuple):
            query,title1,title2 = self.query[batch_index_tuple[0]],self.title[batch_index_tuple[1]],self.title[batch_index_tuple[2]]
            queryvec,title1vec,title2vec = self.convert_to_vec(query), self.convert_to_vec(title1), self.convert_to_vec(title2)
            QueryVec[index_flag,:,:,0]  = queryvec
            Title1Vec[index_flag,:,:,0] = title1vec
            Title2Vec[index_flag,:,:,0] = title2vec
            index_flag+=1
        return QueryVec, Title1Vec, Title2Vec

    def return_init(self):
        self.batch_index_train = 0
        self.batch_index_test  = 0




        


