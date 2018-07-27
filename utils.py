'''
utilities for handling temporal graphs.
These functions or classes are used in ipython notebooks.


'''
import os, re
import datetime

import scipy
import numpy as np
from sklearn.neighbors.kde import KernelDensity

def sec2date(sec):
    '''
    return DD-HH-MM-SS form.
    '''
    day = int(sec//86400)
    hour = int((sec-day*86400)//3600)
    minute = int((sec-day*86400-hour*3600)//60)
    second = int(sec-day*86400-hour*3600-minute*60)
    return {"day":day, "hour":hour, "minute":minute, "second":second}

def unix2date(sec):
    temp = datetime.datetime.fromtimestamp(sec).strftime('%Y-%m-%d %H:%M:%S')
    return temp

def string2timestamp(string):
    '''
    format of string
    2016-11-03T19:37:51.693Z
    '''
    s = string
    return int(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%s'))

def edges_before_timestamp(time2link, to_timestamp, from_timestamp=1157454929):
    '''
    return a list of edges which were linked between from_timestamp and to_timestamp.
    '''
    edges = []
    for key, values in time2link.iteritems():
        if key>=from_timestamp and key<=to_timestamp:
            edges.extend(values)
    return edges

def subedges_in_bins(time2link, binsize=86400, numbin=10, from_timestamp=1157454929):
    '''
    return a list of snapshots which were made in time_bin
    '''
    if from_timestamp > 1157454929:
        subedges = [edges_before_timestamp(time2link,from_timestamp,1157454929)]
    else:
        subedges = []

    for i in range(numbin):
        to_timestamp = from_timestamp + binsize
        subedges.append(edges_before_timestamp(time2link,to_timestamp,from_timestamp) )
        from_timestamp += binsize
    return subedges

def cumulate_subedges(subedges):
    '''
    return a list of cumulative graphs.
    '''
    graphs = []
    for i, _ in enumerate(subedges):
        G = nx.Graph()
        [G.add_edges_from(x) for x in subedges[:i+1]]
        graphs.append(G)
    return graphs


def change_resolution(timestamps, resolution='second'):
    '''
    resolution = day, hour, minute, or second
    [t_i, t_f) => t_f
    '''
    output = []
    for ts in timestamps:
        dt = sec2date(ts)
        day = dt['day']
        hour = dt['hour']
        minute = dt['minute']
        second = dt['second']
        if resolution=='day':
            output.append(day + 1)
        elif resolution=='hour':
            output.append(day*24 + hour + 1)
        elif resolution=='minute':
            output.append(day*24*60 + hour*60 + minute + 1)
        elif resolution=='second':
            output.append(day*24*60*60 + hour*60*60 + minute*60 + second)
        else:
            raise Exception("resolution should be one of {'day', 'hour', 'minute', 'second'}")
        
    return np.array(output)

def get_hist_feature(dict_ids, article_ids, second_threshold, feature='timestamps'):
    '''
    histogram feature. range(0,second_threshold)
    '''
    ts_array = np.zeros((len(article_ids), second_threshold))
    for i, id_ in enumerate(article_ids):
        article_id = id_
        stats = dict_ids[article_id]
        ts_array[i,:] = get_hist_timestamp(stats[feature], second_threshold)

    return ts_array

def get_hist_timestamp(timestamps, threshold):
    '''
    timestamps resolution should be matched with threshold.
    '''
    ts = timestamps
    output = []
    if type(ts).__module__==np.__name__:
        ts = ts.tolist()
        
    for d in range(threshold):
        output.append(ts.count(d))
        
    return np.array(output)
        
def get_article_ids(dict_ids):
    '''
    return list of article ids
    '''
    return dict_ids.keys()

def get_labels(dict_ids, article_ids):
    '''
    return labels
    claim -> 1,    fact_checking ->0
    '''
    output = []
    for id_ in article_ids:
        stats = dict_ids[id_]
        if stats['site_type']=='claim':
            output.append(1)
        elif stats['site_type']=='fact_checking':
            output.append(0)
        elif stats['site_type']=='1':
            output.append(1)
        elif stats['site_type']=='0':
            output.append(0)
        else:
            raise Exception("site_type should be claim or fact_checking")
    return output

def get_linspace_samples(dict_ids, article_ids, nb_samples, 
                         threshold, feature='timestamps', resolution=None,
                         log=True):
    '''
    kde : scipy.stats.gaussian_ked(interval or timestamps)
    linspace : np.linspace(min, max, nb_samples)
    threshold : max of linspace. This unit should be matched with 'resolution'
    resolution : None or 'day','hour','minute'
    '''
    ls_array = np.zeros((len(article_ids), nb_samples))
    for i, id_ in enumerate(article_ids):
        article_id = id_
        stats = dict_ids[id_]
        data = stats[feature]
        if resolution:
            data = change_resolution(data, resolution=resolution)
            data = get_hist_timestamp(data, threshold)
        kde = scipy.stats.gaussian_kde(data)
        if max(data)<threshold:
            t_range = np.linspace(min(data),max(data),nb_samples)
        else:
            t_range = np.linspace(min(data),threshold,nb_samples)
        temp = kde(t_range)
        temp[temp==0] = np.finfo(np.float64).tiny
        ls_array[i,:] = temp
    return np.log(ls_array) if log else ls_array

def get_raw_feature(dict_ids, article_ids, day_threshold, type_features={'day'},
                       feature='timestamps'):
    '''
    day_threshold : cutoff days
    type_features : a set. 'day', 'hour', 'minute', 'second'
    return X which X.shape = (nb_samples, nb_features)
    '''
    nb_day = 0
    nb_hour = 0
    nb_minute = 0
    nb_second = 0
    if 'day' in type_features:
        nb_day = day_threshold
    if 'hour' in type_features:
        nb_hour = day_threshold*24
    if 'minute' in type_features:
        nb_minute = day_threshold*24*60
    if 'second' in type_features:
        nb_second = day_threshold*24*60*60
        
    ts_array = np.zeros((len(article_ids), nb_day+nb_hour+nb_minute+nb_second))
    for i, id_ in enumerate(article_ids):
        article_id = id_
        stats = dict_ids[id_]
        hist_day = []
        hist_hour = []
        hist_minute = []
        hist_second = []
        if nb_day:
            ts_day = change_resolution(stats[feature], 'day')
            hist_day = get_hist_timestamp(ts_day, nb_day)
        if nb_hour:
            ts_hour = change_resolution(stats[feature], 'hour')
            hist_hour = get_hist_timestamp(ts_hour, nb_hour)
        if nb_minute:
            ts_minute = change_resolution(stats[feature], 'minute')
            hist_minute = get_hist_timestamp(ts_minute, nb_minute)
        if nb_second:
            ts_second = stats[feature]
            hist_second = get_hist_timestamp(ts_second, nb_second)

        ts_array[i,:] = np.concatenate((hist_day, hist_hour, hist_minute, hist_second))

    return ts_array

def get_sampled_feature(dict_ids, article_ids, ndim=100,
                        type_features='second', features=['intervals'],
                        bw=1):
    '''
    Each article has 'intervals' (second) between two adjacent retweets.
    ndim : feature dimension. Thus, the number of samplings
    '''
    X = np.zeros((len(article_ids), ndim*len(features)))
    for i, id_ in enumerate(article_ids):
        article_id = id_
        stats = dict_ids[id_]
        for find, feature in enumerate(features):
            raw_data = change_resolution(stats[feature], type_features)
            kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(raw_data.reshape(-1,1))
            X[i,find*ndim:(find+1)*ndim] = np.sort(kde.sample(ndim)[:,0])
    return X

def load_wordembedding(emb_type='glove'):
    '''
    Load word to vector dictionary
    '''
    EMB_FILE = 'glove.6B.100d.txt'
    EMB_PATH = os.path.join('/home/sungyong/workspace/word_embedding/', EMB_FILE)
    EMBEDDING_DIM = int(EMB_FILE.split('.')[2].split('d')[0])
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(EMB_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index
    
def get_avg_wordvector(dict_ids, article_ids, embeddings_index):
    '''
    read each title and get average of word-embeddings
    '''
    emb_dim = len(embeddings_index['the'])
    wv_array = np.zeros((len(article_ids), emb_dim))
    for i, id_ in enumerate(article_ids):
        article_id = id_
        stats = dict_ids[id_]
        title_words = re.sub('[!@#$?,.\'"]', '', stats['title'])
        title_words = title_words.lower().split()
        avg_vector = []
        for word in title_words:
            try:
                embedding_vector = embeddings_index[word]
            except:
                embedding_vector = None
                
            if embedding_vector is not None:
                try:
                    avg_vector = np.vstack([avg_vector, embedding_vector])
                except:
                    avg_vector = embedding_vector
            
        if not len(avg_vector):
            avg_vector = np.zeros((emb_dim,))
                    
        wv_array[i,:] = np.mean(avg_vector, axis=0)
    return wv_array