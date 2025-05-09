from pathlib import Path
import json, sys, os


def update_configs(dataset_folder, config_folder, output_folder=None, mode="train"):

    if output_folder is None:
        # output_folder = Path(config_folder).parent.parent.parent
        output_folder = sys.path[0]
    configs = read_configs()

    log_dir = Path(output_folder) / "logs"
    model_dir = Path(output_folder) / "model"

    if mode == "train":

        cond1 = configs.get("DATA_DIR") == None or configs["DATA_DIR"] != str(
            dataset_folder
        )
        cond2 = configs.get("LOG_DIR") == None or configs["LOG_DIR"] != str(log_dir)
        cond3 = configs.get("MODEL_DIR") == None or configs["MODEL_DIR"] != str(
            model_dir
        )
        cond4 = configs.get("CONFIG_DIR") == None or configs["CONFIG_DIR"] != str(
            config_folder
        )
        if cond1:
            configs.update({"DATA_DIR": str(dataset_folder)})
        if cond2:
            configs.update({"LOG_DIR": str(log_dir)})
        if cond3:
            configs.update({"MODEL_DIR": str(model_dir)})
        if cond4:
            configs.update({"CONFIG_DIR": str(config_folder)})

        if cond1 or cond2 or cond3 or cond4:
            with open(
                f"{sys.path[0]}/DenseConvNet/config/configs.json", "w"
            ) as out_file:
                json.dump(configs, out_file, indent=4)
        else:
            pass
    else:
        cond1 = configs.get("CONFIG_DIR") == None or configs["CONFIG_DIR"] != str(
            config_folder
        )
        # cond2 = configs.get("LOG_DIR") == None or configs["LOG_DIR"] != str(log_dir)

        if cond1:
            configs.update({"CONFIG_DIR": str(config_folder)})
        # if cond2:
        #     configs.update({"LOG_DIR": str(log_dir)})

        if cond1:
            with open(
                f"{sys.path[0]}/DenseConvNet/config/configs.json", "w"
            ) as out_file:
                json.dump(configs, out_file, indent=4)
        else:
            pass

    # with open(f"{sys.path[0]}/DenseConvNet/config/configs.json", "r") as handle:
    #     parsed = json.load(handle)
    # print(json.dumps(parsed, indent=4))


def read_configs():
    config_file_path = f"{sys.path[0]}/DenseConvNet/config/configs.json"

    with open(config_file_path, "r") as f:
        configurations = json.load(f)
    return configurations
