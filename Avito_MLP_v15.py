# This Python 3 environment comes with many helpful analytics libraries installed

import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from contextlib import contextmanager
from operator import itemgetter
import time
from typing import List, Dict
import tqdm

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.sparse import vstack
from nltk.corpus import stopwords

sw = stopwords.words('russian')
#path = '../input/'
path = '/home/vishy/Desktop/Kaggle/Avito/Data/'
subpath = '/home/vishy/Desktop//Kaggle/Avito/Submissions/'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    ex_col = ['item_id', 'user_id', 'deal_probability', 'param_1', 'param_2', 'param_3', 'activation_date']
    df['param'] = (df['param_1'].fillna('') + ' ' + df['param_2'].fillna('') + ' ' + df['param_3'].fillna(''))
    df['description'] = (df['description'].fillna('') +' '+ df['param'].fillna(''))
    del df['param']
    df['description'] = df['description'].str.lower().replace(r"[^[:alpha:]]", " ")
    df['description'] = df['description'].str.replace(r"\\s+", " ")
    df['description_len'] = df['description'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['description_wc'] = df['description'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['description_uniqword'] = df['description'].apply(lambda x: len(set(str(x) for x in x.split())))
    df['desc_words_vs_unique'] = df['description_uniqword'] / df['description_wc'] * 100 # Count Unique Words

    df['title'] = df['title'].str.lower().replace(r"[^[:alpha:]]", " ")
    df['title'] = df['title'].str.replace(r"\\s+", " ")
    df['title_len'] = df['title'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['title_wc'] = df['title'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['title_uniqword'] = df['title'].apply(lambda x: len(set(str(x) for x in x.split())))
    df['title_words_vs_unique'] = df['title_uniqword'] / df['title_wc'] * 100 # Count Unique Words

    #del df['description_wc'], df['description_uniqword'],

    df['image'] = df['image'].map(lambda x: 1 if len(str(x))>0 else 0)
    df["price"] = np.log(df["price"]+0.001)
    df["price"].fillna(-999,inplace=True)
    #df["image_top_1"].fillna('999',inplace=True)
    df["Weekday"] = pd.to_datetime(df['activation_date']).dt.weekday
    col = [c for c in df.columns if c not in ex_col]
    return df[col]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(256, activation='relu')(model_in)
        out = ks.layers.Dropout(0.30)(out)
        out = ks.layers.Dense(128, activation='relu')(out)
        out = ks.layers.Dropout(0.30)(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1, activation='relu')(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=2e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(8), epochs=1, verbose=0)
        return model.predict(X_test, batch_size=2**(8))[:, 0]

def main():
    vectorizer = make_union(
        on_field('description', Tfidf(max_features=100000, stop_words=sw, token_pattern='\w+', norm='l2',
                                    min_df=3, sublinear_tf=True, smooth_idf=False, ngram_range=(1, 2))), #max_df=0.3,
        on_field('title', Tfidf(max_features=100000, stop_words=sw, token_pattern='\w+', norm='l2',
                                    min_df=3, sublinear_tf=True, smooth_idf=False, ngram_range=(1, 2))),
        on_field(['image_top_1','region','category_name','parent_category_name','user_type'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=1)
    with timer('reading data '):
        dtypes = {
        'category_name': 'category',
        'parent_category_name': 'category',
        'region': 'category',
        'item_seq_number': 'uint32',
        'user_type': 'category',
        'image_top_1': 'category',
        'price':'float32',
        'deal_probability': 'float32'
        }
        train = pd.read_csv(path+'train.csv', dtype=dtypes)
        test = pd.read_csv(path+'test.csv', dtype=dtypes)
    with timer('add new features'):
        cat_cols = ['image_top_1','region','city','parent_category_name','category_name','param_1','param_2','param_3','user_type']
        num_cols = ['price', 'deal_probability']
        for c in cat_cols:
            for c2 in num_cols:
                enc = train.groupby(c)[c2].agg(['mean']).astype(np.float32).reset_index()
                enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
                train = pd.merge(train, enc, how='left', on=c)
                test = pd.merge(test, enc, how='left', on=c)
        del(enc)
    with timer('process train'):
        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        y_train = train['deal_probability'].values
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    gc.collect()
    print('train shape',X_train.shape)
    print('valid shape',X_valid.shape)
    with timer('process test'):
        X_test = vectorizer.transform(preprocess(test)).astype(np.float32)
        del test
        gc.collect()
    print('test shape',X_test.shape)

    valid_length = X_valid.shape[0]
    X_valid = vstack([X_valid, X_test])
    del(X_test)
    gc.collect()
    xs = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
    del(X_train, X_valid)
    gc.collect()
    y_pred = fit_predict(xs, y_train=y_train)
    test_pred = y_pred[valid_length:]
    y_pred = y_pred[:valid_length]
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_error(valid['deal_probability'], y_pred))))
    submission = pd.read_csv(path+'test.csv', usecols=["item_id"])
    submission["deal_probability"] = test_pred
    submission['deal_probability'].clip(0.0, 1.0, inplace=True)
    submission.to_csv(subpath+"MLP_V15.csv", index=False)

if __name__ == '__main__':
    main()
