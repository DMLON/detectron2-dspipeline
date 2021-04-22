from detectron2.data import DatasetCatalog, MetadataCatalog
#from load_voc_instances import LoadVOCInstance
from dspipeline.transformers.pipeline import Pipeline
import sys 

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
        self.name=name
        self.dirname=dirname
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
            except:
                e = sys.exc_info()[0]
                stop = True

            if len(dataset) and stop:
                DatasetCatalog.register(
                    self.name,
                    lambda: dataset)

                MetadataCatalog.get(self.name).set(
                    thing_classes=CLASS_NAMES,
                    dirname=self.dirname,
                    split=self.split)

                if self.filter(data):
                    yield True

        

if __name__ == '__main__':
    # print('Testing')
    # load_voc_instances_train = LoadVOCInstance("detectron2-dspipeline/assets/datasets/licenseplates","train")
    # register_data_train = RegisterData("licenseplates_train", "detectron2-dspipeline/assets/datasets/licenseplates", "train")
    # pipeline = load_voc_instances_train | register_data_train
    
    # for i in pipeline.generator():
    #     print(i)

    print('Testing - 2')
    load_voc_instances_train = LoadVOCInstance("detectron2-dspipeline/assets/datasets/licenseplates","train")
    register_data_train = RegisterData("licenseplates_train", "detectron2-dspipeline/assets/datasets/licenseplates", "train")
    load_voc_instances_test = LoadVOCInstance("detectron2-dspipeline/assets/datasets/licenseplates","test")
    register_data_test = RegisterData("licenseplates_test", "detectron2-dspipeline/assets/datasets/licenseplates", "test")

    pipeline = load_voc_instances_train | register_data_train | load_voc_instances_test | register_data_test
    
    for i in pipeline.generator():
        print(i)

    # load_voc_instances_test = LoadVOCInstance("detectron2-dspipeline/assets/datasets/licenseplates","test")
    # register_data_test = RegisterData("licenseplates_test", "detectron2-dspipeline/assets/datasets/licenseplates", "test")
    # pipeline = load_voc_instances_test | register_data_test
    # for i in pipeline.generator():
    #     print(i)