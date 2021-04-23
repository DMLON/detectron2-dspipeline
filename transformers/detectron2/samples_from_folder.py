import random

from dspipeline.transformers.pipeline import Pipeline

class SamplesFromFolder(Pipeline):
    """Pipeline task to get samples from folder."""
    def __init__(self, samples):
        self.samples=samples
    
    def generator(self):
        """Make a list of files."""
        dataset_dict={}
        dataset_dict["files"]=[]
        stop=False
        while self.has_next():
            try:
                data=next(self.source)
                dataset_dict["files"].append(data)
            except StopIteration:
                stop=True

            if len(dataset_dict["files"]) and (len(dataset_dict["files"])==self.samples or stop):
                if self.filter(dataset_dict):
                    yield dataset_dict


