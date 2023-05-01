from os import listdir, chdir, path, system, remove, rmdir
from pprint import pprint
from subprocess import check_output, Popen, PIPE
from . import SCRIPT_HOME
from .config import user, host
import re
def count_lines(file_name):
    """ counts number of lines in a files using 'wc -l' subprocess

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
            edge, congestion = line.split(':')
            edge = edge.strip().strip('()').replace('s', '')
            edge = tuple(edge.split(','))               
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
    system("ffmpeg -r 3 -i %00d.png -vcodec mpeg4 -y {}.mp4".format(movie_name))
    chdir(SCRIPT_HOME)

def test_ssh():
    cmd = "ssh {user}@{host} {cmd}".format(user=user, host=host, cmd='ls -l')
    print(cmd)
    proc = Popen(
        cmd, shell=True, stdout=PIPE, stderr=PIPE)\
        .communicate()
    pprint(proc)

def reindex_down(link:tuple) -> tuple:
    a, b = link
    return (int(a) - 1, int(b) - 1)

def reindex_up(link:tuple) -> tuple:
    a, b = link
    return (int(a) + 1, int(b) + 1)
