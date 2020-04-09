import warnings
warnings.filterwarnings("ignore")
from pyunpack import Archive
import shutil
import math
import time
import pandas as pd
import numpy as np
import scipy
import scipy.sparse
from tqdm import tqdm,tqdm_notebook
from contextlib import contextmanager
import os
import re
import gc
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers
from keras import backend as K

def func_extract_zip(in_path,out_path):
    #ref - https://www.kaggle.com/general/129520
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    Archive(in_path).extractall(out_path)
    for dirname, _, filenames in os.walk(out_path):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            
def func_load_data():
    df_train = pd.read_csv('/kaggle/working/train/train.tsv',sep = '\t')
    df_test = pd.read_csv('/kaggle/working/test_stg2/test_stg2.tsv',sep = '\t')
    return df_train,df_test

def data_preprocess(df):
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]
    
def text_encoder(train,test,vect_type,params):
    vectorizer = CountVectorizer(ngram_range = params[0],min_df = params[1],max_df = params[2],max_features = params[3],token_pattern = '\w+',
                                 dtype = np.float32) if vect_type == 'BOW' else TfidfVectorizer(ngram_range = params[0],min_df = params[1],
                                                                                                max_df = params[2],max_features = params[3],token_pattern = '\w+',
                                                                                                dtype = np.float32)
    train_transform = vectorizer.fit_transform(train)
    test_transform = vectorizer.transform(test)
    feat_names = vectorizer.get_feature_names()
    return train_transform,test_transform
    
def dummy_encoder(train,test):
    train_transform = scipy.sparse.csr_matrix(pd.get_dummies(train[["item_condition_id", 
                                                                         "shipping"]], sparse = True).values)
    test_transform = scipy.sparse.csr_matrix(pd.get_dummies(test[["item_condition_id", 
                                                                         "shipping"]], sparse = True).values)
    return train_transform,test_transform
    
def ridge_model(X_train,y_train,params):
    model = Ridge(solver = "lsqr", fit_intercept=False,alpha = params)
    model.fit(X_train,y_train)
    return model
    
def build_mlp_model1(train_shape):
    model_in = Input(shape=(train_shape,), dtype='float32')
    out = Dense(256, activation='relu')(model_in)
    out = Dense(64, activation='relu')(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(32, activation='relu')(out)
    model_out = Dense(1)(out)
    model = Model(model_in, model_out)
    return model
    
def build_mlp_model2(train_shape):
    model_in = Input(shape=(train_shape,), dtype='float32')
    out = Dense(1024, activation='relu')(model_in)
    out = Dense(512, activation='relu')(out)
    out = Dense(256, activation='relu')(out)
    out = Dense(128, activation='relu')(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(32, activation='relu')(out)
    out = Dense(1)(out)
    model = Model(model_in, out)
    return model
    
def ensemble_generator(preds1,preds2):
    weights = list(np.linspace(0,1,50))
    scores = []
    
    for w in tqdm_notebook(weights):
        preds = (w*preds1) + (1-w)*(preds2)
        scores.append(rmsle_score(df_test_model.price.values,preds))
    
    df_ens = pd.DataFrame({'weights' : weights,'scores':scores})
    w = df_ens.weights[df_ens.scores == min(df_ens.scores)].values[0]
    
    preds_final = (w*preds1) + (1-w)*(preds2)
    return preds_final
    
%%time
if __name__ == "__main__":
    
    # (1) Read Data 
    func_extract_zip('/kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z','/kaggle/working/train/')
    func_extract_zip('/kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip','/kaggle/working/test_stg2/')
    df_train,df_test = func_load_data()
    df_train = df_train[(df_train.price >= 3) & (df_train.price <= 2000)]
    print("(1) done")
    ###################################################################################################################
    
    # (2) Generate Data Encodings
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(np.log1p(df_train['price'].values.reshape(-1, 1)))
    df_train = data_preprocess(df_train)
    df_test = data_preprocess(df_test)

    train_name,test_name = text_encoder(df_train['name'],df_test['name'],'TFIDF',((1,1),1,1.0,100000))
    del df_train['name'],df_test['name']
    train_text,test_text = text_encoder(df_train['text'],df_test['text'],'TFIDF',((1,2),1,1.0,100000))
    del df_train['text'],df_test['text']
    gc.collect()
    train_dummies,test_dummies = dummy_encoder(pd.DataFrame({"shipping" : df_train["shipping"].astype("category"),
                                                             "item_condition_id" : df_train["item_condition_id"].astype("category")}),
                                               pd.DataFrame({"shipping" : df_test["shipping"].astype("category"),
                                                             "item_condition_id" : df_test["item_condition_id"].astype("category")}))
    del df_train['shipping'],df_train['item_condition_id'],df_test['shipping'],df_test['item_condition_id']
    X_train = scipy.sparse.hstack((train_name, train_text, train_dummies)).tocsr().astype('float32')
    X_test = scipy.sparse.hstack((test_name, test_text, test_dummies)).tocsr().astype('float32')
    del train_name,train_text,train_dummies,df_train,test_name,test_text,test_dummies
    gc.collect()
    print("(2) done")
    ###################################################################################################################
    
    # (3) Train Models
    ridge = ridge_model(X_train,y_train,10)
    preds = ridge.predict(X_test)[:, 0]
    preds_ridge = np.expm1(y_scaler.inverse_transform(preds.reshape(-1, 1))[:, 0])
    del ridge,preds
    gc.collect()
    
    mlp1 = build_mlp_model1(X_train.shape[1])
    mlp1.compile(optimizer='adam', loss='mean_squared_error')
    for i in range(2):
        mlp1.fit(X_train,y_train,batch_size = 2**(8+i),epochs = 1,verbose = 1)
    preds = mlp1.predict(X_test,verbose = 1)[:, 0]
    preds_mlp1 = np.expm1(y_scaler.inverse_transform(preds.reshape(-1, 1))[:, 0])
    del mlp1,preds
    gc.collect()
    
    mlp2 = build_mlp_model2(X_train.shape[1])
    mlp2.compile(optimizer='adam', loss='mean_squared_error')
    for i in range(2):
        mlp2.fit(X_train,y_train,batch_size = 2**(8+i),epochs = 1,verbose = 1)
    preds = mlp2.predict(X_test,verbose = 1)[:, 0]
    preds_mlp2 = np.expm1(y_scaler.inverse_transform(preds.reshape(-1, 1))[:, 0])
    del mlp2,preds
    gc.collect()
    ###################################################################################################################
    
    # (4) Generate Best Ensemble
    preds_mlps = 0.55*preds_mlp1 + 0.45*preds_mlp2
    preds_f = 0.1*preds_ridge + 0.9*preds_mlps
    ###################################################################################################################
    
    # (6) Submissions output file
    submission = pd.DataFrame({'test_id' : df_test.index.values,'price' : preds_f})
    submission.to_csv("submission.csv", index=False)
    print('(6) done')
    ###################################################################################################################
