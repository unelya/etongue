from pyteomics import mzxml
from collections import defaultdict, namedtuple
from typing import Tuple, List, Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from src.represent_tea_sample import DataEncodeMode
from src.init_params import CYCLES, TARGET, N_CLASSES, N_PEAKS

MSMetadata = namedtuple('Metadata', 'cls, sample, m')

Y_SCALE = 10**5

def onehot(target, n_classes):
    zeros = np.zeros(n_classes)
    zeros[target - 1] = 1
    return zeros

class MSDataSample:
    """Class representation of data in one experiment file"""
    
    @staticmethod
    def extract_metadata_from_fname(fname: str) -> MSMetadata:
        samplename = fname.rstrip('.mzXML')
        name, sample, number = samplename.split('_')[0], samplename.split('_')[1], samplename.split('_')[-1]
        numlist = number.split('~')
        first_digit = 10*int(numlist[0])
        if len(numlist) > 1:
            second_digit = int(number[-1])
        else:
            second_digit = 0
        cls = int(name[-1])
        sample = int(sample)
        m = first_digit + second_digit
        return MSMetadata(cls, sample, m)
    
    @staticmethod
    def parse_data_from_file(path: Path):
        #rt = [[],[]]
        current = [[],[]]
        mz = [[],[]]
        intensity =[[],[]]
        with mzxml.read(path) as reader:
            for data in reader:
                if data['polarity'] == '+':
                    #rt[0].append(data['retentionTime']) - not needed for simple ms measurements
                    current[0].append(data['totIonCurrent'])
                    mz[0].append(list(data['m/z array']))
                    intensity[0].append(list(data['intensity array']))
                elif data['polarity'] == '-':
                    #rt[1].append(data['retentionTime'])
                    current[1].append(data['totIonCurrent'])
                    mz[1].append(list(data['m/z array']))
                    intensity[1].append(list(data['intensity array']))
            
        return np.array(current), mz, intensity
    
    @staticmethod
    def best_ms(current, mz, intensity, polarity) -> List[np.array]:
        def extract_max_current(pol_current, pol_mz, pol_intensity) -> np.ndarray:
            n = np.argmax(current)
            return np.array(pol_mz[n]), np.array(pol_intensity[n])
        
        if polarity == '+':
            ms = np.array(extract_max_current(current[0], mz[0], intensity[0]))
        elif polarity == '-':
            ms = np.array(extract_max_current(current[1], mz[1], intensity[1]))

        return ms
    
    def to_nn_sample(self, 
                     mode=DataEncodeMode.MS,
                     onehot_target=False,
                     vector=False):
        poz_ms = MSDataSample.best_ms(self.current, self.mz, self.intensity, '+')
        neg_ms = MSDataSample.best_ms(self.current, self.mz, self.intensity, '-')
        
        max_poz_ind = np.argsort(-np.array(poz_ms[1]))[:N_PEAKS]
        max_neg_ind = np.argsort(-np.array(neg_ms[1]))[:N_PEAKS]
        
        poz_x = poz_ms[0][max_poz_ind]
        poz_y = poz_ms[1][max_poz_ind] / Y_SCALE
        
        poz_y = poz_y[np.argsort(poz_x)]
        poz_x = np.sort(poz_x)
        
        neg_x = neg_ms[0][max_neg_ind]
        neg_y = neg_ms[1][max_neg_ind] / Y_SCALE
        
        neg_y = neg_y[np.argsort(neg_x)]
        neg_x = np.sort(neg_x)
        
        result = (poz_x, poz_y, neg_x, neg_y)
        if vector:
            result = np.hstack(result)
        else:
            result = np.vstack(result).transpose()
        
        target = self.target
        if onehot_target:
            target = onehot(target, N_CLASSES)
            
        return {TARGET: target, CYCLES: result}
        
        return result
    
    def represent(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        poz_ms = MSDataSample.best_ms(self.current, self.mz, self.intensity, '+')
        print(poz_ms.shape[1])
        neg_ms = MSDataSample.best_ms(self.current, self.mz, self.intensity, '-')
        print(neg_ms.shape[1])
        
        plt.rcParams.update({'font.size': 14})
        
       
        
        axs[0].set_title('Positive ions')
        axs[1].set_title('Negative ions')
        axs[0].plot(poz_ms[0], poz_ms[1]/100000)
        axs[1].plot(neg_ms[0], neg_ms[1]/100000)
        
        nn = self.to_nn_sample()[CYCLES].transpose()
        
        axs[0].scatter(nn[0], nn[1], s=15, c='r')
        axs[1].scatter(nn[2], nn[3], s=15, c='r')
        for i in range(2):
            axs[i].set_xlabel('m/z (Da)')
            axs[i].set_ylabel('Intensity')
        
        df_cm = pd.DataFrame(nn, range(4), range(N_PEAKS))
        plt.figure(figsize=(16,4))
        plt.title("NN representation")
        sn.heatmap(df_cm, annot=True, fmt= ".1f", annot_kws={"size": 10})
        
    def __init__(self, file):
        self.metadata: Metadata = MSDataSample.extract_metadata_from_fname(file)
        self.current, self.mz, self.intensity = MSDataSample.parse_data_from_file(file)
    
    @property
    def key(self) -> Tuple[int]:
        return tuple([self.metadata.cls, self.metadata.sample, self.metadata.m])
    
    @property
    def cls(self) -> int:
        return self.metadata.cls
    
    @property
    def target(self) -> int:
        return self.cls




