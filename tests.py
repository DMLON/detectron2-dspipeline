from dspipeline.transformers.utils.files import list_files_in_txt

for i in list_files_in_txt('detectron2-dspipeline/assets/datasets/licenseplates','train'):
    print(i)
