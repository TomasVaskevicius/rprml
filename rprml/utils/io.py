import os
import pickle


def create_directories(fname):
    """ Given a full path to some file, creates all relevant directories.
    For example, given ./a/b/c/run.pickle will create directories ./a/b/c/.
    Note that given ./a/b/c this function will only create directories
    ./a/b/ assuming that c is a file name. """
    path = '/'.join(fname.split('/')[:-1])
    os.makedirs(path, exist_ok=True)


def save_to_disk(obj, fname):
    full_path = fname + '.pickle'
    create_directories(full_path)
    with open(full_path, 'wb') as _file:
        pickle.dump(obj, _file)


def load_from_disk(full_path):
    with open(full_path, 'rb') as _file:
        obj = pickle.load(_file)
    return obj
