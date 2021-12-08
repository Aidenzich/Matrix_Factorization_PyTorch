import pandas as pd
import time
import json
import math
from tqdm import tqdm

def dtype_memory_usage(dataframe, dtype_list=['float','int','object']):
    for dtype in dtype_list:
        selected_dtype = dataframe.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print(f'Average memory usage for {dtype} columns: {mean_usage_mb:03.2f} MB')

def memory_usage(pandas_obj):
    r"""A Quick method to read pandas_obj used memory.
    """
    if isinstance(pandas_obj, pd.DataFrame):
        usage_byte = pandas_obj.memory_usage(deep=True).sum()
    else: 
        usage_byte = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_byte / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)

def get_cat2id(pandas_series):
    r"""Convert `pandas_series` to a catgories base id dictionary.
    Usually used with below method id2cat.
    Returns
    ========
    dict
        cat2idx dictionary
    list
        catsid list
    """
    cats = pandas_series.astype('category').cat.codes
    cat_dict = dict(zip(pandas_series, cats))
    return cat_dict, cats

def id2cat(cat_dict, idx):
    return list(cat_dict.keys())[list(cat_dict.values()).index(idx)]

def timer(func):
    def wrapper( *args , **kwargs ):
        s = time.perf_counter()
        v = func( *args , **kwargs )
        e = time.perf_counter()
        print(f"{func.__name__} takes {e-s} s.")
        return v
    return wrapper

def readjson2dict(filename):
    filename = filename + ".json"
    with open(filename) as jf:
        json_dict = json.load(jf)
    return json_dict

def savedict2json(dict, filename):
    filename = filename + ".json"
    with open(filename, 'w') as jf:        
        json.dump(dict, jf)
    return filename

def update2json(update, filename):
    try:
        with open(filename, 'r+') as jf:
            o_dict = json.load(jf)
    except:
        o_dict = {}
    o_dict[time.time()] = update
    with open(filename, "w") as jf:
        json.dump(o_dict, jf)

def recall_evaluate(predict_dict, real_dict):
    recall3_sum = 0
    recall5_sum = 0
    recall10_sum = 0
    evaluatelen = len(real_dict)
    for i in tqdm(real_dict):
        real_list = real_dict[i]
        if len(real_list)==0:
            evaluatelen -=1
            continue

        try:
            predict_list = predict_dict[i]
        except:
            predict_list = []
        recall3 = len( list( set(predict_list[:3]) & set(real_list) )) / len(real_list)
        recall5 = len( list( set(predict_list[:5]) & set(real_list) )) / len(real_list)
        recall10 = len( list( set(predict_list[:10]) & set(real_list) )) / len(real_list)
        recall3_sum += recall3
        recall5_sum += recall5
        recall10_sum += recall10
    result = {
        "recall@3": recall3_sum/evaluatelen,
        "recall@5": recall5_sum/evaluatelen,
        "recall@10": recall10_sum/evaluatelen,
    }
    return result


def smooth_user_preference(x):
    """normalized values
    """
    return math.log(1+x, 2)