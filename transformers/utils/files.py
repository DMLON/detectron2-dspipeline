from fvcore.common.file_io import PathManager
import numpy as np
import os

def walk_to_level(path, level=None):
    """
    Returns a generator which gives the root folder, directories and files
    in the given path

    Parameters
    ----------
    path: Path to folder

    level: How many levels deep should it go ?
    """
    if level is None:
        yield from os.walk(path)
        return

    path = path.rstrip(os.path.sep)
    num_sep = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            # When some directory on or below the desired level is found, all
            # of its subdirs are removed from the list of subdirs to search next.
            # So they won't be walked.
            del dirs[:]


def list_files(path, valid_exts=None, level=None):
    """
    Returns a generator which gives the files names in the folder

    Parameters
    ----------
    path: Path to folder

    valid_exts: only yield the extensions
    """
    # Loop over the input directory structure
    for (root_dir, dir_names, filenames) in walk_to_level(path, level):
        for filename in sorted(filenames):
            # Determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            if valid_exts and ext.endswith(valid_exts):
                # Construct the path to the file and yield it
                file = os.path.join(root_dir, filename)
                yield file


def list_files_in_txt(path,filename):
    """
    Returns a generator which gives the lines of the given txt path file

    Parameters
    ----------
    path: Path to file

    filename: Name of the file without .txt
    """
    with open(os.path.join(os.path.curdir,path, filename + ".txt"),"r") as f:
        fileids=np.loadtxt(f, dtype=np.str)
    
    for fileid in fileids:
        yield fileid


if __name__ == '__main__':
    print("path:")
    
    # path = os.path.join(f'{os.path.curdir}/dspipeline/assets/datasets/licenseplates','train.txt')
    # with open(path,"r") as f:
    #     fileids=np.loadtxt(f, dtype=np.str)
    
    # for fileid in fileids:
    #     print(fileid)

    for i in list_files_in_txt('dspipeline/assets/datasets/licenseplates','train'):
        print(i)
    