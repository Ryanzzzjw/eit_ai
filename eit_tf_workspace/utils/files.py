from enum import Enum

################################################################################
# Standards file extention
################################################################################

class FileExt(Enum):
    """Set all files extension used

    to get the str-value use eg.:
    FileExt.mat.value
    f'{FileExt.mat}'
    print(FileExt.mat)
    
    Args:
        Enum ([type]): [description]
    """
    mat= '.mat' # matlab files
    pkl= '.pkl' # pickle files
    txt= '.txt' # text files

    def __repr__(self):
      return self.value

    def __str__(self):
        return str(self.value)


if __name__ == "__main__":
    from glob_utils.log.log  import main_log
    main_log()
    f=f'f {FileExt.txt}'
    print(f,  FileExt.mat )