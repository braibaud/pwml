import os as os
import numpy as np
import pickle as pk
import hashlib as hl


def md5(filepath):
    hash_md5 = hl.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_ndarray(filepath, data):
    np.savez_compressed(
        file=filepath, 
        x=data)
    return filepath

def load_ndarray(filepath):
    np.load(
        file=filepath,
        allow_pickle=True)['x']


def save_cache_dict_file(filepath, dict):
    return save_ndarray(
        filepath=filepath,
        data=np.array(dict, dtype=object))

def load_cache_dict_file(filepath):
    return load_ndarray(
        filepath=filepath)[()]


def load_from_npz(filepath):
    with np.load(filepath, allow_pickle=True) as npz_file:
        data = dict(npz_file.items())
    return data

def save_to_npz(filepath, data_dict):
    np.savez_compressed(
        filepath,
        **data_dict)
    return filepath


def save_to_pickle(filepath, data):
    with open(filepath, 'wb') as fp:
        pk.dump(data, fp, protocol=pk.HIGHEST_PROTOCOL)
    return filepath


def load_from_pickle(filepath):
    with open(filepath, 'rb') as fp:
        data = pk.load(fp)
    return data


def save_to_file(filepath, text):
    with open(filepath, 'w') as fp:
        fp.write(text)
    return filepath


def get_timestamped_filename(filename, extension, stamp):
    return '{0}-{1}.{2}'.format(
        filename,
        get_stamp_text(stamp),
        extension)


def get_timestamped_name(name, stamp):
    return '{0}-{1}'.format(
        name,
        get_stamp_text(stamp))


def get_stamp_text(stamp):
    return '{0:04d}-{1:02d}-{2:02d}-{3:02d}-{4:02d}-{5:02d}'.format(
        stamp.year,
        stamp.month,
        stamp.day,
        stamp.hour,
        stamp.minute,
        stamp.second)


def create_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_stamp_folder(directory, stamp):
    return create_if_not_exists(
        os.path.join(
            directory,
            get_stamp_text(stamp)))
