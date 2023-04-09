import pickle as pk

with open('filename.pickle', 'rb') as handle:
    X_test = pk.load(handle)
    y_test = pk.load(handle)