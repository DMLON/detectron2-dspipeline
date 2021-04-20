import os

from detectron2.config import get_cfg
from detectron2.engine import default_setup

from dspipeline.transformers.pipeline import Pipeline

class SetConfig(Pipeline):
    """Pipeline task to set configurations."""
    def __init__(self,args):
        """
        Parameters
        ----------
        args : argparser
            Args from terminal.
        """
        self.args=args

        super(SetConfig,self).__init__()

    def map(self,data):
        """Register data in detectron2."""
        cfg = get_cfg()
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        if self.args.outputdir is not None:
            cfg.OUTPUT_DIR = self.args.outputdir
        cfg.freeze()
        default_setup(cfg, self.args)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        return cfg

    def filter(self,data): #Dataset is registered => set config
        if data is True:
            return True
        else:
            return False