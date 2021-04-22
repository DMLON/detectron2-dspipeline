import cv2
import sys
import argparse

from dspipeline.processor import Processor

from transformers.detectron2.load_test_images import LoadTestImages
from transformers.detectron2.register_data import RegisterData
from transformers.detectron2.set_config import SetConfig
from transformers.detectron2.samples_from_catalog import SamplesFromCatalog

from transformers.detectron2.predictor import Predictor
from transformers.detectron2.display_predictions import DisplayPredictions


def parse_args():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Licenseplates prediction")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)

    # Detectron settings
    ap.add_argument("--config-file",
                    required=True,
                    help="path to config file")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="minimum score for instance predictions to be shown (default: 0.5)")
    ap.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                    help="modify model config options using the command-line")

    return ap.parse_args()



def main(args):

    if args.confidence_threshold is not None:
        # Set score_threshold for builtin models
        args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))
        args.opts.append('MODEL.RETINANET.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))

    # Create pipeline steps
    dataset_name = "licenseplates_test"

    load_test_images=LoadTestImages("detectron2-dspipeline/assets/datasets/licenseplates","test")
    register_data_test = RegisterData(dataset_name, "detectron2-dspipeline/assets/datasets/licenseplates", "test")
    set_config=SetConfig(args)
    samples_from_catalog=SamplesFromCatalog(dataset_name, args.samples)
    predictor=Predictor()
    display_predictions=DisplayPredictions(args.scale, dataset_name)

    # Create image processing pipeline

    pipeline = (
        load_test_images  |
        register_data_test  |
        set_config  |
        samples_from_catalog   |
        predictor   |
        display_predictions
    )

     # Create processor for processing pipeline
    process=Processor(pipeline)
    try:
        process.run(verbose=True)
    except:
        e = sys.exc_info()[0]
        return
    finally:
        print(f"[INFO] Finalizing process [{process.id}]...")

    

    cv2.destroyAllWindows()


if __name__ == "__main__":
    #args = parse_args()    #Enable when the script is running through terminal
    #print("Command Line Args:", args)
    print(os.path.abspath)
    args=argparse.Namespace(
        config_file='detectron2-dspipeline/configs/lp_faster_rcnn_R_50_FPN_3x.yaml',
        samples=3,
        scale=0.8,
        confidence_threshold=0.85,
        opts=['MODEL.WEIGHTS', 'detectron2-dspipeline/output/model_final.pth']
         )   # Disable when run through terminal'''

    main(args)