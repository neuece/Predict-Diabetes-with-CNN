# build word2vec using medical claims
# WORD2VEC is trained with different part of data becaused limited RAM resource

# comments are removed on purpose. 

import gensim
from gensim.models import word2vec
import logging
import numpy as np
import urllib.request
import os
import zipfile
import pandas as pd
import glob
import datetime
import pickle
import argparse
from collections import Counter
import gc
from datetime import datetime
import math
from ast import literal_eval
import random

df = pd.read_csv('../cms_data/Medical_CODES_MED1.csv',index_col=None, header=0)
df.columns = ['DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT','MONTH_WINDOW','COMB_CODE']


df['COMB_CODE'] = df['COMB_CODE'].apply(literal_eval)
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[str(i) for i in x])
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:random.sample(x,len(x)))

test = pd.DataFrame()
test = df['COMB_CODE'].apply(list)
t1=test.values.T.tolist()
dict_all = pickle.load(open('../cms_data/all_dict.pickle', 'rb'))

x = list(range(1,20802))
x = [str(i) for i in x]
x = random.sample(x,len(x))
x1 = x[20800:20803]
x3 = x[:20800]
x3[-1]
x2 = [x3[i:i+10] for i in range(0, len(x3), 10)]
t1 = t1+x2
t1.append(x1)


model = word2vec.Word2Vec(t1, iter=15, min_count=1, size=200, window=7, workers=6)

model.save('../cms_data/' + "CODE_VEC_W7")
vocab_size = len(model.wv.vocab)
print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])
print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])


df = pd.DataFrame()
test = pd.DataFrame()
df = pd.read_csv('../cms_data/Medical_CODES_MED2.csv',index_col=None, header=0)
df.columns = ['DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT','MONTH_WINDOW','COMB_CODE']
df['COMB_CODE'] = df['COMB_CODE'].apply(literal_eval)
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[str(i) for i in x])
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:random.sample(x,len(x)))
test = df['COMB_CODE'].apply(list)
t2=test.values.T.tolist()

model.build_vocab(t2, update=True)
model.train(t2, total_examples=model.corpus_count, epochs=model.iter)
model.save('../cms_data/' + "CODE_VEC_W7_STEP2")


df = pd.DataFrame()
test = pd.DataFrame()
df = pd.read_csv('../cms_data/Medical_CODES_MED3.csv',index_col=None, header=0)
df.columns = ['DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT','MONTH_WINDOW','COMB_CODE']
df['COMB_CODE'] = df['COMB_CODE'].apply(literal_eval)
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[str(i) for i in x])
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:random.sample(x,len(x)))
test = df['COMB_CODE'].apply(list)
t3=test.values.T.tolist()

model.build_vocab(t3, update=True)
model.train(t3, total_examples=model.corpus_count, epochs=model.iter)
model.save('../cms_data/' + "CODE_VEC_W7_STEP3")


df = pd.DataFrame()
test = pd.DataFrame()
df = pd.read_csv('../cms_data/Medical_CODES_MED4.csv',index_col=None, header=0)
df.columns = ['DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT','MONTH_WINDOW','COMB_CODE']
df['COMB_CODE'] = df['COMB_CODE'].apply(literal_eval)
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[str(i) for i in x])
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:random.sample(x,len(x)))
test = df['COMB_CODE'].apply(list)
t4=test.values.T.tolist()

model = word2vec.Word2Vec.load('../cms_data/' + "CODE_VEC_W7_STEP3")
model.build_vocab(t4, update=True)
model.train(t4, total_examples=model.corpus_count, epochs=model.iter)
model.save('../cms_data/' + "CODE_VEC_W7_STEP4")


# load IP data

df = pd.DataFrame()
test = pd.DataFrame()
df = pd.read_csv('../cms_data/Medical_CODES_IP.csv',index_col=None, header=0)
df.columns = ['DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT','MONTH_WINDOW','COMB_CODE']
df['COMB_CODE'] = df['COMB_CODE'].apply(literal_eval)
df['CLM_FROM_DT'] = df['CLM_FROM_DT'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
df['CLM_THRU_DT'] = df['CLM_THRU_DT'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[i for i in x if i!=0])
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[str(i) for i in x])
df['LoS'] = (df['CLM_THRU_DT']-df['CLM_FROM_DT']).dt.days+1
# OPTION1: repeat rows (w/o removing zeros)
df = df.loc[df.index.repeat(df['LoS'])]
df.reset_index(drop=True,inplace=True)
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:random.sample(x,len(x)))
test = df['COMB_CODE'].apply(list)
t5=test.values.T.tolist()

model = word2vec.Word2Vec.load('../cms_data/' + "CODE_VEC_W7_STEP4")
model.build_vocab(t5, update=True)
model.train(t5, total_examples=model.corpus_count, epochs=model.iter)
model.save('../cms_data/' + "CODE_VEC_W7_STEP5")

# load OP data

df = pd.DataFrame()
test = pd.DataFrame()
df = pd.read_csv('../cms_data/Medical_CODES_OP1.csv',index_col=None, header=0)
df.columns = ['DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT','MONTH_WINDOW','COMB_CODE']
df = df[df['COMB_CODE'].notnull()]
df['COMB_CODE'] = df['COMB_CODE'].apply(literal_eval)
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[str(i) for i in x])
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:random.sample(x,len(x)))
test = df['COMB_CODE'].apply(list)
t6=test.values.T.tolist()

#model = word2vec.Word2Vec.load('../cms_data/' + "CODE_VEC_W7_STEP5")
model.build_vocab(t6, update=True)
model.train(t6, total_examples=model.corpus_count, epochs=model.iter)
model.save('../cms_data/' + "CODE_VEC_W7_STEP6")


df = pd.DataFrame()
test = pd.DataFrame()
df = pd.read_csv('../cms_data/Medical_CODES_OP2.csv',index_col=None, header=0)
df.columns = ['DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT','MONTH_WINDOW','COMB_CODE']
df = df[df['COMB_CODE'].notnull()]
df['COMB_CODE'] = df['COMB_CODE'].apply(literal_eval)
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:[str(i) for i in x])
df['COMB_CODE'] = df['COMB_CODE'].apply(lambda x:random.sample(x,len(x)))
test = df['COMB_CODE'].apply(list)
t7=test.values.T.tolist()

model = word2vec.Word2Vec.load('../cms_data/' + "CODE_VEC_W7_STEP6")
model.build_vocab(t7, update=True)
model.train(t7, total_examples=model.corpus_count, epochs=model.iter)
model.save('../cms_data/' + "CODE_VEC_W7_STEP7")