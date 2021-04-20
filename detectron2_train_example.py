import os
import argparse
from processor import Processor

from licenseplates.config import setup_cfg
from detectron2.engine import  launch
from detectron2.engine import default_argument_parser

from transformers.detectron2.load_voc_instances import LoadVOCInstance
from transformers.detectron2.register_data import RegisterData
from transformers.detectron2.train_model import TrainModel
from transformers.detectron2.set_config import SetConfig


def parse_args():

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Image processing pipeline")
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image files")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-os", "--out-summary", default="summary.json",
                    help="output JSON summary file name")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="face detection batch size")

    return ap.parse_args()
                  

def main(args):
    # Create pipeline steps

    load_voc_instances = LoadVOCInstance("assets/datasets/licenseplates","train")
    register_data_train = RegisterData("licenseplates_train", "assets/datasets/licenseplates", "train")
    register_data_test = RegisterData("licenseplates_train", "assets/datasets/licenseplates", "train")

    set_config=SetConfig(args)
    train_model=TrainModel()
    # Create image processing pipeline
    pipeline = (
        load_voc_instances |
        register_data_train |
        register_data_test |
        set_config  |
        train_model
    )
    
    # Create processor for processing pipeline
    process=Processor(pipeline)
    try:
        process.run(verbose=True)
    except:
        return
    finally:
            print(f"[INFO] Finalizing process [{process.id}]...")

if __name__ == "__main__":
    #args = parse_args()    # Disable during debugging 
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )