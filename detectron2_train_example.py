import os
import argparse
from dspipeline.processor import Processor

from detectron2.engine import  launch
from detectron2.engine import default_argument_parser

import sys

from transformers.detectron2.load_voc_instances import LoadVOCInstance
from transformers.detectron2.register_data import RegisterData
from transformers.detectron2.train_model import TrainModel
from transformers.detectron2.set_config import SetConfig

import dspipeline as ds

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
    load_voc_instances_train = LoadVOCInstance("detectron2-dspipeline/assets/datasets/licenseplates","train")
    register_data_train = RegisterData("licenseplates_train", "detectron2-dspipeline/assets/datasets/licenseplates", "train")
    load_voc_instances_test = LoadVOCInstance("detectron2-dspipeline/assets/datasets/licenseplates","test")
    register_data_test = RegisterData("licenseplates_test", "detectron2-dspipeline/assets/datasets/licenseplates", "test")
    set_config=SetConfig(args)
    train_model=TrainModel(args)


    # Create image processing pipeline
    pipeline = (
        load_voc_instances_train |
        register_data_train |
        load_voc_instances_test  |
        register_data_test |
        set_config  |
        train_model
    )

    pipeline_train = (load_voc_instances_train | register_data_train)
    pipeline_test = (load_voc_instances_test  | register_data_test)
    pipeline_model = (set_config  | train_model)
    
    # Create processor for processing pipeline
    process=Processor(pipeline)
    #process_train=Processor(pipeline_train)
    #process_test=Processor(pipeline_test)
    #process_model=Processor(pipeline_model)
    try:
        #process_train.run(verbose=True)
        #process_test.run(verbose=True)
        #process_model.run(verbose=True)
        process.run(verbose=True)
    except:
        e = sys.exc_info()[0]
        return
    finally:
        print(f"[INFO] Finalizing process [{process.id}]...")

if __name__ == "__main__":
    #args = parse_args()    # Disable during debugging 
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    args=argparse.Namespace(
        config_file='detectron2-dspipeline/configs/lp_faster_rcnn_R_50_FPN_3x.yaml',
         dist_url='tcp://127.0.0.1:50152', 
         eval_only=False, 
         machine_rank=0, 
         num_gpus=1, 
         num_machines=1, 
         opts=[], 
         resume=False,
         outputdir='./detectron2-dspipeline/output'
         )   # Disable when run through terminal
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )