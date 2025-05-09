import json, os, sys
from pathlib import Path


def create_output_folders(out_folder):

    if out_folder is None:
        out_folder = sys.path[0]

    log_folder = Path(out_folder) / "logs"
    log_folder.mkdir(exist_ok=True, parents=True)

    model_folder = Path(out_folder) / "model"
    model_folder.mkdir(exist_ok=True, parents=True)

    s_pred_folder = Path(out_folder) / "logs/predictions/sagittal"
    s_pred_folder.mkdir(exist_ok=True, parents=True)

    c_pred_folder = Path(out_folder) / "logs/predictions/coronal"
    c_pred_folder.mkdir(exist_ok=True, parents=True)

    a_pred_folder = Path(out_folder) / "logs/predictions/axial"
    a_pred_folder.mkdir(exist_ok=True, parents=True)

    # agg_pred_folder = Path(out_folder) / "logs/predictions/combined"
    # agg_pred_folder.mkdir(exist_ok=True, parents=True)
