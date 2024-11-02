from src.init_params import EXCLUDE_CLASSES, EXCLUDE_ELECTRODES, CLASSES_MAPPING

from collections import defaultdict, namedtuple
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np

Metadata = namedtuple('Metadata', 'cls, sample, sensor, kcl, mat')

class DataSample:
    """Class representation of data in one experiment file"""
    
    @staticmethod
    def extract_metadata_from_fname(fname: str) -> Metadata:
        stem = fname.rstrip('.txt')
        if 'TEA' in stem:
            stem = stem.lstrip('TEA')
            cls = int(stem[0])
            stem = stem[1:]
            if stem[0] == 'T':
                stem = stem[2:]
            else:
                stem = stem[1:]
        else:
            cls = 0
            stem = stem.lstrip('WATER.')
        
        if cls in EXCLUDE_CLASSES:
            cls = -1
        else:
            cls = CLASSES_MAPPING[cls]
        
        sample, sensor, kcl, _, mat = stem.split('_')
        sample = int(sample)
        sensor = int(sensor)
        kcl = int(kcl)
        mat = mat.lower()
        if mat in EXCLUDE_ELECTRODES:
            mat = -1
        
        return Metadata(cls, sample, sensor, kcl, mat)
    
    @staticmethod
    def parse_data_from_file(path: Path) -> Tuple[np.ndarray]:
        """Parse 3 cycles from file"""
        
        def parse_cycle(raw_cycle: str) -> List[List[float]]:
            rows = raw_cycle.split('\n')[1:]
            rows = [[float(value) for value in row.split()] for row in rows]
            return rows
        
        raw_data = path.read_text()
        parts = raw_data.split('\n\n')
        raw_cycles = parts[1], parts[3], parts[5]
        cycles = tuple([np.array(parse_cycle(raw_cycle)) for raw_cycle in raw_cycles])
            
        return cycles
    
    @staticmethod
    def split_cycle(cycle: np.ndarray) -> np.ndarray:
        """Split contuguous cycle measurement into full front and back runs"""
        
        index_max = np.argmax(cycle[:, 1])
        index_min = np.argmin(cycle[:, 1])
        back = cycle[index_max:index_min]
        front = np.concatenate((cycle[index_min:], cycle[:index_max]), axis=0)
        return front, back
        
    def __init__(self, path: Path):
        self.metadata: Metadata = DataSample.extract_metadata_from_fname(path.name)
        self.cycles = DataSample.parse_data_from_file(path)
        self.cycle1, self.cycle2, self.cycle3 = self.cycles
        
    @property
    def target(self) -> Tuple[int]:
        return self.metadata.cls
    
    @property
    def cls(self) -> str:
        _cls = self.metadata.cls
        return "WATER" if _cls == 0 else f"TEA{_cls}"
    
    @property
    def key(self) -> Tuple[int]:
        return tuple([self.metadata.cls, self.metadata.sample, 
                      self.metadata.sensor, self.metadata.kcl])