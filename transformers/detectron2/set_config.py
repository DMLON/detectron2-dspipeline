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
        """Set config on detectron2."""
        if data==True:
            data={}
        else:
            data=data

        cfg = get_cfg()
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)

        if "outputdir" in self.args:
            if self.args.outputdir is not None:
                cfg.OUTPUT_DIR = self.args.outputdir
            
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            
        cfg.freeze()
        default_setup(cfg, self.args)
        
        data["cfg"]=cfg
        return data

        