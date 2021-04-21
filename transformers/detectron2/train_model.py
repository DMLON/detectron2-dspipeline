from dspipeline.transformers.pipeline import Pipeline

from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from .utils.evaluator import VOCDetectionEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return VOCDetectionEvaluator(dataset_name)


class TrainModel(Pipeline):
    """Pipeline task to register VOC annotations."""
    def __init__(self,args):
        """
        Init TrainModel instance
        """
        self.args=args
        super(TrainModel,self).__init__()

    def map(self,data):
        """Train model specified in config."""
        if self.args.eval_only:
            model = Trainer.build_model(data["cfg"])
            DetectionCheckpointer(model, save_dir=data["cfg"].OUTPUT_DIR).resume_or_load(
                data["cfg"].MODEL.WEIGHTS, resume=self.args.resume
            )
            res = Trainer.test(data["cfg"], model)
            return res

        trainer = Trainer(data["cfg"])
        trainer.resume_or_load(resume=self.args.resume)
        return trainer.train()