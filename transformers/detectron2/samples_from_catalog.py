import random

from dspipeline.transformers.pipeline import Pipeline

from detectron2.data import DatasetCatalog

class SamplesFromCatalog(Pipeline):
    """Pipeline task to get samples from catalog."""
    def __init__(self,dataset_name,samples):
        """
        Parameters
        ----------
        dataset_name : str
            Dataset's name.
        samples : int
            Number of samples to get.
        """
        self.samples=samples
        self.dataset_name=dataset_name

        super(SamplesFromCatalog,self).__init__()

    def map(self,data):
        """Yields the image content and annotations."""

        dict={}
        dict["cfg"]=data
        dict["files"]=[]

        dataset=DatasetCatalog.get(self.dataset_name)

        for d in random.sample(dataset, self.samples):
            dict["files"].append(d)
        
        return dict


