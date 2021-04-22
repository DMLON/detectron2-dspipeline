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

    ap.add_argument("--dataset-name",default="dataset_name",
                    help="name of the dataset")

    ap.add_argument("-os", "--out-summary", default="summary.json",
                    help="output JSON summary file name")

    ap.add_argument("--batch-size", type=int, default=1,
                    help="face detection batch size")

    return ap.parse_args()
                  

def main(args):

    split_train='train'
    split_val='test'    #TODO: Poner de nombre 'val'

    dataset_train_name=args.dataset_name+'_'+split_train
    dataset_val_name=args.dataset_name+'_'+split_val
    

    # Create pipeline steps
    # TODO: Etapa para hacer train_test_split???
    # TODO: Tener automatizado para, mediante un arg, saber que Loader usar (VOC, COCO, etc...)
    load_voc_instances_train = LoadVOCInstance(args.input,split_train)
    register_data_train = RegisterData(dataset_train_name, args.input, split_train)
    load_voc_instances_test = LoadVOCInstance(args.input,split_val)
    register_data_test = RegisterData(dataset_val_name, args.input, split_val)
    set_config=SetConfig(args)
    train_model=TrainModel(args)


    # Create model train pipeline
    pipeline = (
        load_voc_instances_train |
        register_data_train |
        load_voc_instances_test  |
        register_data_test |
        set_config  |
        train_model
    )

    # Create processor for processing pipeline
    process=Processor(pipeline) #TODO: Agregar un summary del process

    try:
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
        input="detectron2-dspipeline/assets/datasets/licenseplates",
        dataset_name='licenseplates',
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