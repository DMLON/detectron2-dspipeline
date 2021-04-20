import os
import argparse
from processor import Processor

from transformers.load_images import LoadImage
from transformers.resize_image import ResizeImage
from transformers.save_image import SaveImage


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

    load_images = LoadImage(args.input)
    resize_image = ResizeImage(width=28, height=28)
    save_image = SaveImage(args.output)

    # Create image processing pipeline
    pipeline = (
        load_images |
        resize_image |
        save_image
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
    args = parse_args()    # Disable during debugging 
    # args=argparse.Namespace(
    #     input="assets/images", 
    #     output="output",
    #     out_summary="output.json",
    #     batch_size=1
    #     )   # Disable when run through terminal

    main(args)