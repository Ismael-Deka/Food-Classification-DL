import pickle as pk

with open('test.pickle', 'rb') as handle:
    test_dataset = pk.load(handle)
    test_loader = pk.load(handle)
    