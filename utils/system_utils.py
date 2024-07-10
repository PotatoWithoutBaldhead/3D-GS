from errno import EEXIST
import os


def mkdir_p(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(folder_path):
            pass
        else:
            raise
        
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)