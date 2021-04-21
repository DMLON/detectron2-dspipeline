import sys
import cv2

from detectron2.engine import DefaultPredictor

from dspipeline.transformers.pipeline import Pipeline

def get_dataset(dict):
    return dict

class Predictor(Pipeline):
    """Pipeline task to predict from input."""
    def __init__(self):
        """
        Class constructor.
        """

        super(Predictor,self).__init__()

    def generator(self):
        """Make a prediction from model specified in config."""
        data=next(self.source)
        predictor=DefaultPredictor(data["cfg"])
        
        for f in data["files"]:
            img = cv2.imread(f)
            prediction = predictor(img)
            result={
                "image":img, 
                "prediction":prediction
                }
            if self.filter(result):
                yield self.map(result)



