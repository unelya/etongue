import tensorflow as tf
from math import ceil
from collections import defaultdict, namedtuple
from typing import Tuple, List, Dict, Optional
import numpy as np

from src.represent_tea_sample import CompoundDataSample, DataEncodeMode
from src.init_params import CYCLES, TARGET, Y_MEAN, Y_STD

def preprocess(dataset_elem, y):
        dataset_elem[CYCLES] = (dataset_elem[CYCLES] - Y_MEAN) / Y_STD
        return dataset_elem, y
    
def create_dataset(
    samples_list: List[CompoundDataSample],
    batch_size: int,
    mode: DataEncodeMode = DataEncodeMode.FIRST_CYCLE_ONLY,
    augmentations = [],
    to_tf=True,
    onehot_target=False,
    vector=False,
    do_preprocess=True,
) -> tf.data.Dataset:
    X = []
    y = []
    for cds in samples_list:
        nn_input = cds.to_nn_sample(
            mode=mode,
            onehot_target=onehot_target,
            vector=vector)
        x = nn_input[CYCLES]
        for aug in augmentations:
            x = aug(x)
        X.append(x)
        y.append(nn_input[TARGET])
            
    if to_tf:
        X = {CYCLES: X}
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if do_preprocess:
            dataset = dataset.map(preprocess)
        dataset = dataset.shuffle(batch_size * 2).batch(batch_size)
    else:
        dataset = X, y
    return dataset

def split_dataset(samples_list: List, 
                  test_size: float):
    """
    Here we use by-label stratification: since number of samples is small,
    default uniform shuffle will produce train/test split that is not uniform
    by class. Such stratification allows to have [test_size] proportion 
    of every label in test and 1 - [test_size] proportion of every
    label in train.
    """
    by_target = defaultdict(list)
    for sample in samples_list:
        by_target[sample.target].append(sample)
    for k in by_target:
        np.random.shuffle(by_target[k])
    
    train = []
    test = []
    for target in by_target.values():
        l = int(len(target) * test_size)
        train.extend(target[l:])
        test.extend(target[:l])
    
    return train, test
        
#Data augmentation

def scale_aug(max_factor):
    """Scale y axis by a random coefficient in [1 - max_factor, 1]"""
    def scale(x):
        factor = tf.random.uniform((1,), minval=1.0-max_factor, maxval=1.0, dtype=tf.dtypes.float32)
        return x * factor
    return scale

aug = scale_aug(0.2)

def make_data(data_train, data_val, **kwargs):
    train = create_dataset(data_train, **kwargs)
    kwargs["augmentations"] = []
    kwargs["batch_size"] = 1
    val = create_dataset(data_val, **kwargs)
    return train, val

#tf.random.set_seed(42)
#np.random.seed(42)