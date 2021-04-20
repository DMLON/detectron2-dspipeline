from detectron2.data import DatasetCatalog, MetadataCatalog

from ..pipeline import Pipeline

CLASS_NAMES = [
    "licenseplate",
]

class RegisterData(Pipeline):
    """Pipeline task to register VOC annotations."""
    def __init__(self,name,dirname,split):
        """
        Parameters
        ----------
        name : str
            name of the dataset.
        dirname : str
            path to "annotations" and "images".
        split : str
            one for "train" and one for "test"
        """
        self.dirname=name
        self.split=dirname
        self.split=split

        super(RegisterData,self).__init__()

    def generator(self):
        """Register data in detectron2."""
        dataset=[]
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream
                data = next(self.source)
                dataset.append(data)
            except StopIteration:
                stop = True

            if len(dataset) and stop:
                DatasetCatalog.register(
                    self.name,
                    lambda: get_dataset(dataset))

                MetadataCatalog.get(self.name).set(
                    thing_classes=CLASS_NAMES,
                    dirname=self.dirname,
                    split=self.split)

                if self.filter(data):
                    yield True

        def get_dataset(dict):
            return dict

