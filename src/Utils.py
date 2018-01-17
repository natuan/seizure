import numpy as np

# Check if two dictionaries are equal
def dicts_equal(dict1, dict2):
    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())
    if not np.array_equal(keys1, keys2):
        print("Keys not matched")
        return False
    for k in keys1:
        vals1 = dict1[k]
        vals2 = dict2[k]
        if not np.array_equal(vals1, vals2):
            print("Values at key {} not matched".format(k))
            return False
    return True

