from src.init_params import N_ELECTRODES, MATERIAL_ORDER, TARGET, CYCLES, X_MIN, X_MAX, N_BINS, N_CLASSES
from src.represent_cv_sample import DataSample

from collections import defaultdict, namedtuple
from enum import Enum
from typing import Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

def grid_mean(x, y, x_min, x_max, n_bins):
    assert len(x) == len(y)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    sums = np.histogram(x, bins, weights=y)[0]
    counts = np.histogram(x, bins)[0]
    new_y = np.divide(sums, 
                      counts, 
                      out=np.zeros_like(sums), 
                      where=counts!=0)
    return (bins[1:] + bins[:-1]) / 2, new_y

class DataEncodeMode(Enum):
    FIRST_CYCLE_ONLY = "FCO"
    SECOND_CYCLE_ONLY = "SCO"
    ALL_CYCLES = "AC"
    
    MS = "MS"

class CompoundDataSample:
    """Class representation of complete data sample used to predict target"""
    
    @staticmethod
    def order_samples(samples: List[DataSample]):
        assert set([s.metadata.mat for s in samples]) == set(MATERIAL_ORDER)
        sample_dict = {sample.metadata.mat: sample for sample in samples}
        return [sample_dict[mat] for mat in MATERIAL_ORDER]
    
    def __init__(self, samples: List[DataSample]) -> None:
        assert len(samples) == N_ELECTRODES
        common_key = samples[0].key
        assert all([sample.key == common_key for sample in samples])
        common_cls = samples[0].cls
        assert all([sample.cls == common_cls for sample in samples])
        self.key = common_key
        self.cls = common_cls
        self.samples = CompoundDataSample.order_samples(samples)
        
    @property
    def target(self):
        return self.samples[0].metadata.cls
    
    def get_cycle(self, electrode_index: int, cycle_index: int) -> None:
        sample = self.samples[electrode_index]
        cycle = sample.cycles[cycle_index]
        
        front, back = DataSample.split_cycle(cycle)
        xf = front[:, 1]
        yf = front[:, 2]
        xb = back[:, 1]
        yb = back[:, 2]
        return xf, yf, xb, yb
        
    def represent(self) -> None:
        plt.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(N_ELECTRODES, 3, figsize=(15, 15))
        d = self.key
        fig.suptitle(f'{self.cls}: smpl {d[1]} snsr {d[2]} KCL {d[3]}')
        fig.tight_layout(pad=3.0)
        for j in range(N_ELECTRODES):
            meta: Metadata = self.samples[j].metadata
            for i in range(3):
                if meta.mat == 'cu':
                    subheader = 'Cu-BTC'
                elif meta.mat == 'fe':
                    subheader = 'MIL-100'
                elif meta.mat == 'zn':
                    subheader = 'ZIF-8'
                elif meta.mat == 'ni':
                    subheader = 'Ni-BTC'
                axs[j, i].set_title(subheader + f', cycle {i}')
                xf, yf, xb, yb = self.get_cycle(j, i)
                axs[j, i].plot(xf, 1000000*yf, 'b', alpha=0.6)
                axs[j, i].plot(xb, 1000000*yb, 'b', alpha=0.6)
                axs[j, i].grid()
                axs[j, i].set_xlabel('Voltage, V')
                axs[j, i].set_ylabel('Current, μA')       
            
    def to_nn_sample(self, 
                     mode: DataEncodeMode = DataEncodeMode.FIRST_CYCLE_ONLY,
                     vector=False,
                     onehot_target=True) \
            -> Dict[str, np.ndarray]:
        if onehot_target:
            target = onehot(self.target, N_CLASSES)
        else:
            target = self.target
        
        cycles = []
        for sample in self.samples:
            if mode == DataEncodeMode.FIRST_CYCLE_ONLY:
                c_range = [sample.cycle1]
            elif mode == DataEncodeMode.SECOND_CYCLE_ONLY:
                c_range = [sample.cycle2]
            else:
                c_range = sample.cycles
                
            for cycle_pair in c_range:
                for cycle in DataSample.split_cycle(cycle_pair):
                    y = cycle[:, 2]
                    x = cycle[:, 1]
                    x, y_averaged = grid_mean(x, y, X_MIN, X_MAX, N_BINS)
                    cycles.append(y_averaged)
            
        if vector:
            cycles = np.hstack(cycles)
        else:
            cycles = np.vstack(cycles).transpose()
        return {TARGET: target, CYCLES: cycles}        

def onehot(target, n_classes):
    zeros = np.zeros(n_classes)
    zeros[target] = 1
    return zeros
