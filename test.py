import pickle as pk

with open('test.pickle', 'rb') as handle:
    X_test = pk.load(handle)
    y_test = pk.load(handle)