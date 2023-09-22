from os import listdir, chdir, makedirs, path, system, remove, rmdir
from pprint import pprint
import shutil
from subprocess import check_output, Popen, PIPE
from time import process_time
from ..constants import SCRIPT_HOME
from .config import user, host
import re

from onset.constants import PLOT_DIR

def clock(f, *args, **kwargs):
    # clocks the time to run a function
    # return (result, time) tuple 
    start = process_time()
    result = f(*args, **kwargs)
    end = process_time()        
    return result, (end-start)

def count_lines(file_name):
    """counts number of lines in a files using 'wc -l' subprocess

    Args:
        file_name (str): file to count

    Returns:
        int: number of lines in the file
    """
    return int(check_output(["wc", "-l", file_name]).split()[0])


def get_edge_congestion(edge_congestion_file) -> dict:
    edge_congestion = {}
    core_link_regex = re.compile("^\s*\(s[0-9]+,s[0-9]+\)")
    with open(edge_congestion_file) as ecf:
        for line in ecf:
            if not core_link_regex.match(line):
                continue
            edge, congestion = line.split(":")
            edge = edge.strip().strip("()").replace("s", "")
            edge = tuple(edge.split(","))
            congestion = float(congestion.strip())
            edge_congestion[edge] = congestion

    return edge_congestion


def cast_pair_to_int(a, b):
    return int(a), int(b)


def delete_dir(d):
    # Delete all files of directory d, and d itself.
    # If the directory doesn't exist nothing happens.
    try:
        contents = listdir(d)
        for c in contents:
            remove(path.join(d, c))
        rmdir(d)

    except FileNotFoundError:
        pass


def export_vid(folder):
    chdir(folder)
    movie_name = path.basename(folder)
    system(
        "ffmpeg -r 3 -i %00d.png -vcodec mpeg4 -y {}.mp4".format(movie_name)
    )
    chdir(SCRIPT_HOME)


def test_ssh():
    cmd = "ssh {user}@{host} {cmd}".format(user=user, host=host, cmd="ls -l")
    print(cmd)
    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    pprint(proc)


def reindex_down(link: tuple) -> tuple:
    a, b = link
    return (int(a) - 1, int(b) - 1)


def reindex_up(link: tuple) -> tuple:
    a, b = link
    return (int(a) + 1, int(b) + 1)


def clear_dir(dir_path):
    """
    Clears directory.
    Input:  dir_path: directory to clear.
    """
    for content in listdir(dir_path):
        content_path = path.join(dir_path, content)
        if path.isfile(content_path):
            # print(f'Removing: {file_path}')
            remove(content_path)
        elif path.isdir(content_path):
            shutil.rmtree(content_path)


def make_dir(directory):
    """
    Create directory if doesn't already exist.
    """
    if not path.exists(directory):
        makedirs(directory)


def postfix_str(str_in, postfix):
    if "." in str_in:
        s = str_in.split(".")
        prefix = "".join(s[:-1])
        extension = s[-1]
        new_str = f"{prefix}_{postfix}.{extension}"
    else:
        new_str = f"{str_in}_{postfix}"
    return new_str


def save_raw_data(X, Y, outfile="data", xlabel="", ylabel=""):
    base_file, _ = path.splitext(path.basename(outfile))
    final_outfile = path.join(PLOT_DIR, base_file)
    with open(final_outfile + ".csv", "w") as fob:
        if xlabel and ylabel:
            fob.write(f"{xlabel},{ylabel}\n")
        else:
            fob.write("X,Y\n")
        for x, y in zip(X, Y):
            fob.write(f"{x},{y}\n")

    return


def key_to_json(data):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, (tuple, frozenset)):
        return str(data)
    raise TypeError


def to_json(data):
    if data is None or isinstance(
        data, (bool, int, float, tuple, range, str, list)
    ):
        return data
    if isinstance(data, (set, frozenset)):
        return sorted(data)
    if isinstance(data, dict):
        return {key_to_json(key): to_json(data[key]) for key in data}
    raise TypeError

