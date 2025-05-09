# Model parameters

EPOCHS = 150
NUM_CLASS = 3
BATCH_SIZE = 8
EPSILON = 0.0001
STEPS_PER_EPOCH = 30
INPUT_SHAPE = {
    "sagittal": (312, 256, 1),
    "coronal": (256, 256, 1),
    "axial": (256, 312, 1),
}

# Plane indices

PLANE_IDX = {"sagittal": 0, "coronal": 1, "axial": 2}
COMP_PLANE_IDX = {"sagittal": (1, 2), "coronal": (0, 2), "axial": (0, 1)}

# Folder paths

# DATA_DIR = "/Users/souvik/Neurogazer/projects/dataset/S3_backup"
DATA_DIR = "/Volumes/Seagate_backup/datasets"
LOG_DIR = "/Users/souvik/Neurogazer/projects/segmentation/brain_extraction/DenseConvNet/run_18/logs"
MODEL_DIR = "/Users/souvik/Neurogazer/projects/segmentation/brain_extraction/DenseConvNet/run_18/model"

# DATA_DIR = "/home/ubuntu/projects/data/S3_data"
# LOG_DIR = "/home/ubuntu/projects/brain_extraction/DenseConvNet/run_7/logs"
# MODEL_DIR = "/home/ubuntu/projects/brain_extraction/DenseConvNet/run_6/model"
