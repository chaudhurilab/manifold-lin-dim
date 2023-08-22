# October 22nd 2015
# Modifying these. I think that the ECC stuff uses some of these but called from a local directory
# so should be fine.
# Actually haven't modified these yet but should. Rename load_file and save_file to
# mention text, and then add binary functions.
# Some general functions for housekeeping, loading and saving
# etc.

from __future__ import division
import pickle
import glob


def load_file_from_pattern(file_pattern):
    file_matches = glob.glob(file_pattern)
    if len(file_matches) > 1:
        print('Multiple matches. Using the first one')
    if len(file_matches) == 0:
        print('No file found')
        return
    fname = file_matches[0]
    data = load_pickle_file(fname)
    return data, fname


def load_pickle_file(filename):
    fr = open(filename, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data


def save_pickle_file(data, filename):
    fw = open(filename, 'wb')
    pickle.dump(data, fw)
    fw.close()
    return 1
