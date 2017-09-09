import os
import six
import inspect
from tqdm import tqdm


def make_dir_one(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dir(path):
    separated_path = path.split('/')
    tmp_path = ''
    for directory in separated_path:
        tmp_path = tmp_path + directory + '/'
        if directory == '.':
            continue
        make_dir_one(tmp_path)
    return True


def find_files(path, affix_flag=False):
    if path[-1] == '/':
        path = path[:-1]
    if affix_flag is False:
        return [path + '/' + name for name in os.listdir(path)]
    else:
        return [name for name in os.listdir(path)]


def remove_slash(path):
    return path[:-1] if path[-1] == '/' else path


def create_progressbar(end, desc='', stride=1, start=0):
    if type(end) is int or type(end) is float:
        return tqdm(six.moves.range(int(start), int(end), int(stride)), desc=desc, leave=False)
    else:
        return tqdm(end, desc=desc, leave=False)


# store builtin print
old_print = print


def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)


# globaly replace print with new_print
inspect.builtins.print = new_print
