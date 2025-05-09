from pathlib import Path
from silence_tensorflow import silence_tensorflow
import os, sys, pathlib, argparse
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage.transform import resize

cwd = os.getcwd()
os.chdir(sys.path[0])
# os.chdir(Path(__file__).parent)
silence_tensorflow()


from DenseConvNet.file_utils.config_utils import update_configs
from DenseConvNet.file_utils.helper_functions import create_output_folders


def run_model(input_path, config_path, output_path, mode="train"):
    update_configs(input_path, config_path, output_path, mode=mode)

    if mode == "train":

        create_output_folders(output_path)
        from DenseConvNet.scripts.train import Train_Model

        training = Train_Model(input_path, config_path, output_path)
        training.train("sagittal")
        training.train("coronal")
        training.train("axial")
        print(
            "       ................................................................DONE................................................................"
        )

    elif mode == "predict":
        from DenseConvNet.scripts.predict import MakePredictions

        prediction = MakePredictions(input_path, output_path)
        image, map1, f1 = prediction.predict(save=False)
        return image, map1, f1
    else:
        raise ValueError("mode can be one of 'train', 'predict'")

    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", default="predict", help=" mode can be train/[predict]"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=f"{sys.path[0]}/DenseConvNet/config/configs.json",
        help=" absolute path to config file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=f"{sys.path[0]}",
        help=" absolute path to output file (for predict) or folder (for train)",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="input volume for predict, dataset directory for train",
    )
    args = parser.parse_args()

    mode = args.mode
    DATA_PATH = Path(args.input)
    CONFIG_PATH = Path(args.config)
    OUTPUT_PATH = Path(args.output)

    # DATA_PATH = Path("x")
    # DATA_PATH = Path("/Volumes/Seagate_backup/datasets")
    # DATA_PATH = Path(
    #     "/Users/souvik/Desktop/test/subjects/R_AH0320_KRAA_000624_01/brain_extraction/T1w_acpc.nii.gz"
    # )
    # CONFIG_PATH = Path(
    #     "/Users/souvik/Neurogazer/projects/segmentation/brain_extraction/DenseConvNet/run_21/DenseConvNet/config/configs.json"
    # )
    # OUTPUT_PATH = Path("/Users/souvik/Desktop/test")
    # OUTPUT_PATH = Path(
    #     "/Users/souvik/Neurogazer/projects/segmentation/brain_extraction/DenseConvNet/run_20"
    # )
    # OUTPUT_PATH = Path(
    #     "/Users/souvik/Desktop/test/subjects/R_AH0313_KRAA_000616_01/brain_extraction/predicted.nii.gz"
    # )
    # OUTPUT_PATH = Path(
    #     "/Volumes/Seagate_backup/datasets/Neurogazer/val/R_AH0320_KRAA_000631_01/brain_extraction"
    # )
    img, map1, pred_mask1 = run_model(DATA_PATH, CONFIG_PATH, OUTPUT_PATH, mode=mode)
    mask1 = nib.load(
        "/Users/souvik/Desktop/test/subjects/R_AH0320_KRAA_000624_01/brain_extraction/T1w_acpc_brain_mask.nii.gz"
    )
    mask1 = nib.processing.conform(
        mask1, orientation="LAS", out_shape=(256, 312, 256), voxel_size=(0.7, 0.7, 0.7)
    ).get_fdata()
    print(np.sum(map1[120], axis=-1))

    # n = np.random.choice(np.arange(60, 200))
    n = 120
    lists = [
        img[n],
        img[:, n],
        img[:, :, n],
        mask1[n],
        mask1[:, n],
        mask1[:, :, n],
        # np.ones_like(img[:, :, n]),
        # map1[n, :, :, 0],
        # map1[n, :, :, 1],
        # map1[n, :, :, 2],
        # np.abs(mask1[n] - map1[n, :, :, 2]),
        # np.abs(mask1[:, n] - map1[:, n, :, 2]),
        # np.abs(mask1[:, :, n] - map1[:, :, n, 2]),
        pred_mask1[n],
        pred_mask1[:, n],
        pred_mask1[:, :, n],
    ]

    fig, axs = plt.subplots(3, 3, figsize=(10, 12))
    for j, i in enumerate(axs.flatten()):
        i.imshow(lists[j], cmap="gray")
        i.set_xticks([])

        i.set_yticks([])

    plt.tight_layout()
    plt.show()
