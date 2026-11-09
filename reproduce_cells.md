# Reproduction cells (Google Colab, T4 GPU)

Run these cells in order in a new Colab notebook
(Runtime → Change runtime type → T4 GPU).

## 1. Clone the repository
```python
%cd /content
!rm -rf PrivateKT
!git clone https://github.com/diptikale-IoT/PrivateKT.git
!mkdir -p /content/PrivateKT/logs /content/PrivateKT/results
```

## 2. Port to TensorFlow 2 and define `run_privatekt`
```python
%cd /content/PrivateKT/Code

# TF2 port (safe to run more than once)
!sed -i 's/from keras.utils.np_utils import to_categorical/from tensorflow.keras.utils import to_categorical/' preprocessing.py PrivateKT.py
!sed -i 's/^from keras.layers import \*/from tensorflow.keras.layers import */; s/^from keras.models/from tensorflow.keras.models/; s/^from keras.optimizers import \*/from tensorflow.keras.optimizers import SGD/; s/SGD(lr=lr)/SGD(learning_rate=lr)/' model.py
!grep -q "verbose=0" evaluation.py || sed -i 's/\.predict(\([^)]*\))/.predict(\1, verbose=0)/' PrivateKT.py evaluation.py

import numpy as np, random, time, csv, importlib, tensorflow as tf
import preprocessing, model, evaluation, PrivateKT as pkt
for m in (preprocessing, model, evaluation, pkt): importlib.reload(m)
from preprocessing import load_data, partition_public_dataset, local_data_partition
from model import get_model
from evaluation import evaluation as evaluate
from PrivateKT import calculate_beta, KnowledgeBuffer, PrivateKT
print("GPU:", tf.config.list_physical_devices('GPU'))

# Authors' hyper-parameters (unchanged): eps=5, K=2, buffer B=25, lr=0.05,
# public data ratio 0.2, Dirichlet alpha=0.5
def run_privatekt(seed, T, log_every=1):
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    tr_x, tr_y = load_data('../demo_data/', 'train'); te_x, te_y = load_data('../demo_data/', 'test')
    tr_x, tr_y, pub = partition_public_dataset(tr_x, tr_y, 0.2)
    users = local_data_partition(tr_y, alpha=0.5)
    beta = calculate_beta(5, 2, tr_y.shape[1]); kb = KnowledgeBuffer(25); m = get_model(0.05)
    times = []
    with open(f'../results/seed{seed}.csv', 'w', newline='') as f, open(f'../logs/seed{seed}.txt', 'w') as lg:
        w = csv.writer(f); w.writerow(['seed', 'round', 'accuracy', 'seconds'])
        for i in range(T):
            t0 = time.time(); PrivateKT(kb, m, users, tr_x, tr_y, pub, beta, 2)
            acc = evaluate(m, te_x, te_y); dt = time.time() - t0; times.append(dt)
            w.writerow([seed, i, acc, round(dt, 2)]); f.flush()
            line = f'seed {seed} round {i} acc {acc:.4f} time {dt:.1f}s'
            lg.write(line + '\n'); lg.flush()
            if i % log_every == 0 or i == T - 1: print(line)
    return times
```

## 3. Run 3 seeds (T = 10 rounds each, as in this reproduction)
```python
T = 10
for s in (0, 1, 2):
    run_privatekt(seed=s, T=T, log_every=1)
```

## 4. Combine results and draw the figure
```python
import pandas as pd
%cd /content/PrivateKT
df = pd.concat([pd.read_csv(f'results/seed{s}.csv') for s in (0, 1, 2)])
df.to_csv('results/my_results.csv', index=False)
!python make_figure.py
```

Expected output: `results/seed0.csv`–`seed2.csv`, `results/my_results.csv` and `fig1.pdf`.
Note: name the function `run_privatekt`, not `run`, because a bare `run(...)` in Colab triggers the `%run` magic.
