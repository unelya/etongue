N_ELECTRODES = 4
N_CLASSES = 6
EXCLUDE_ELECTRODES = []
N_ELECTRODES = N_ELECTRODES - len(EXCLUDE_ELECTRODES)

X_MIN = -1.2
X_MAX = 1.2
N_BINS = 50

EXCLUDE_CLASSES = {0}

keys = [i for i in range(N_CLASSES) if i not in EXCLUDE_CLASSES]
values = [i for i in range(N_CLASSES - len(EXCLUDE_CLASSES))]
CLASSES_MAPPING = {i: j for i, j in zip(keys, values)}

N_CLASSES = N_CLASSES - len(EXCLUDE_CLASSES)

all_materials = ['cu', 'ni', 'zn', 'fe']
for mat in EXCLUDE_ELECTRODES:
    all_materials.remove(mat)
MATERIAL_ORDER = all_materials
MATERIAL_ORDER

TARGET = "target"
CYCLES = "cycles"

batch_size = 8

Y_MEAN = -0.0000027965327204708562
Y_STD = 0.000024358655771268833

N_PEAKS = 10