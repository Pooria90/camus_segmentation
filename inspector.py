import os, sys
import pickle
import numpy as np

run_id = sys.argv[1] if len(sys.argv) > 1 else 'v4-run-3'

with open(f'./results/{run_id}/args.pkl','rb') as f:
    args = pickle.load(f)

with open(f'./results/{run_id}/logs.pkl','rb') as f:
    logs = pickle.load(f)

print ('args:\n')
for k,v in vars(args).items():
    print(k,' : ', v)

print ('\nlog keys:\n',logs.keys())

ii = np.argmax(logs['dice_metric'])
print (f'\nBest Metric obtained at epoch {logs["epoch"][ii]}:\n',logs['dice_mean'][ii])
