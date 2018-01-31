import os

import pandas as pd

TRAIN_FILE_HDF = '../input/train.hdf'
TRAIN_FILE_CSV = '../input/train.csv'
TEST_FILE = '../input/test.csv'


def load_train(nrows=None, save_to_hdf=False):
    """Loading of the train file.

    Parameters
    ----------
    path : string
        The path of the file to open.

    nrows : int or None, optional (default=None)
        Number of rows to read. If None, all the file is read.

    save_to_hdf : bool, optional (default=True)
        If the open file is a csv, save it as hdf for a faster loading.

    Returns
    -------
    data : pd.Dataframe, shape = [n_samples, n_features]
        Return the data sample.
    """
    return fast_load_data(path=TRAIN_FILE_HDF, nrows=nrows, save_to_hdf=save_to_hdf)


def load_test(nrows=None, save_to_hdf=False):
    """Loading of the test file.

    Parameters
    ----------
    path : string
        The path of the file to open.

    nrows : int or None, optional (default=None)
        Number of rows to read. If None, all the file is read.

    save_to_hdf : bool, optional (default=True)
        If the open file is a csv, save it as hdf for a faster loading.

    Returns
    -------
    data : pd.Dataframe, shape = [n_samples, n_features]
        Return the data sample.
    """
    return fast_load_data(path=TEST_FILE, nrows=nrows, save_to_hdf=save_to_hdf)


def fast_load_data(path, nrows=None, save_to_hdf=False):
    """Fast loading of a data file.

    Parameters
    ----------
    path : string
        The path of the file to open.

    nrows : int or None, optional (default=None)
        Number of rows to read. If None, all the file is read.

    save_to_hdf : bool, optional (default=True)
        If the open file is a csv, save it as hdf for a faster loading.

    Returns
    -------
    data : pd.Dataframe, shape = [n_samples, n_features]
        Return the data sample.
    """
    filename, file_extension = os.path.splitext(path)
    load_hdf = True
    
    if file_extension == '.hdf':
        if not os.path.exists(path):
            load_hdf = False
    elif file_extension == '.csv':
        load_hdf = False
    elif file_extension == '':
        return fast_load_data(filename+'.hdf', nrows, save_to_hdf)
    else:
        raise NameError("Wrong file extension.")
    
    if load_hdf:
        data = pd.read_hdf(filename+'.hdf', key='data', stop=nrows)
    else:
        data = pd.read_csv(filename+'.csv', nrows=nrows)

    if not load_hdf and save_to_hdf:
        data.to_hdf(filename+'.hdf', 'data')

    return data
