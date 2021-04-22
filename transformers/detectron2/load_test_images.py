
import os
from dspipeline.transformers.pipeline import Pipeline

from dspipeline.transformers.utils.files import list_files_in_txt

class LoadTestImages(Pipeline):
    """Pipeline task to get paths to images."""
    def __init__(self,dirname,split):
        """
        Parameters
        ----------
        dirname : str
            path to "annotations" and "images".
        split : str
            path to "annotations" and "images" txt files.
        """
        self.dirname=dirname
        self.split=split

        super(LoadTestImages,self).__init__()

    def generator(self):
        """Yields paths to images."""

        source=list_files_in_txt(self.dirname, self.split)
        while self.has_next():
            try:              
                fileid=next(source)
                jpeg_file = os.path.join(self.dirname, "images", fileid + ".jpg")
                data = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                }
                if self.filter(data):
                    yield self.map(data)

            except StopIteration:
                return

        
                    


         

