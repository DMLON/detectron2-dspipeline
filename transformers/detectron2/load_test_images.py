
from dspipeline.transformers.pipeline import Pipeline

from dspipeline.transformers.utils.files import list_files

class LoadTestImages(Pipeline):
    """Pipeline task to get paths to images."""
    def __init__(self,path,valid_exts=(".jpg", ".png"),level=None):
        """
        Parameters
        ----------
        path : str
           Path to test images folder.
        """
        self.path=path
        self.valid_exts=valid_exts
        self.level=level

        super(LoadTestImages,self).__init__()

    def generator(self):
        """Yields paths to images."""
        data={}
        data["files"]=[]
        stop=False
        source=list_files(self.path,self.valid_exts,self.level)
        while self.has_next():
            try:
                img=next(source)
                data["files"].append(img)
            except StopIteration:
                stop=True
        
            if len(data["files"]) and stop:
                if self.filter(data):
                    yield self.map(data)
        
                    


         

