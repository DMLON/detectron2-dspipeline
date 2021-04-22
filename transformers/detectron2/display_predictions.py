import cv2
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog
from dspipeline.transformers.pipeline import Pipeline

class DisplayPredictions(Pipeline):
    """Pipeline task to display images and bboxs."""
    def __init__(self,scale,dataset_name):
        """
        Parameters
        ----------
        scale : int
            scale.
        dataset_name : str
            Dataset's names.
        """
        self.scale=scale
        self.dataset_name=dataset_name

        super(DisplayPredictions,self).__init__()

    def map(self,data):
        """Display images with bbox."""
        
        visualizer = Visualizer(data["image"][:, :, ::-1],
                                metadata=MetadataCatalog.get(self.dataset_name),
                                scale=self.scale)
        vis = visualizer.draw_instance_predictions(data["prediction"]["instances"].to("cpu"))
        cv2.imshow(self.dataset_name, vis.get_image()[:, :, ::-1])

        # Exit? Press ESC
        if cv2.waitKey(0) & 0xFF == 27:
            return data

        return data

