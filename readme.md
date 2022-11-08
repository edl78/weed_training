# Training repo for obdb

## Architecture
- Depends on having a mongodb instance with all annotation data collected from CVAT.
- Bayesisan optimization via Sherpa for hyperparameters.
- Will output a number of networks ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] or any combination of pytorch available networks.
- Highly adaptable, adjust the dataset class with you own data

![](doc_img/architecture.png)

## T-SNE based train/val splits
- use the analytics functionality to get information on the clusters produced by t-sne to have a fair representation of features in both train and val datasets


## How to run
- Shell scripts for building and running the container. These are build_training.sh and run_training.sh
- After starting the contrainer, go to /code and run `python3 torch_model_runner.py -f path_to_json_settings_file` defaults to /code/settings_file.json 
- The trained networks can then be found in the mapped folder train or /train in the container. A file with optimal training parameters is also located together with the network.
- Watch the hyper parameter tuning on localhost:8880 and the training and validation losses for all runs on localhost:6006 after pointing your local tensorboard to code/runs/name_of_the_run
- The json configuration file looks like this: 
```json
{
    "variants": ["resnet18","resnet34"],
    "resnet18": {
        "run_hpo": 0,
        "optimal_hpo_settings": {
            "gamma": 0.03376820338738614,
            "lr": 3.934578332944719e-05,
            "momentum": 0.3317994388235662,
            "step_size": 3,
            "weight_decay": 0.5559432320028943
        }
    },
    "resnet34": {
        "run_hpo": 1
    },
    "resnet50": {
        "run_hpo": 1
    },            
    "max_plateau_count": 20,
    "search_epochs": 10,
    "num_initial_data_points": 5,
    "max_num_trials": 16,
    "annotations_list": ["FieldData 20200520145736 1L GH020068",
                        "FieldData 20200515101008 3R GH070071",
                        "FieldData 20200603102414 1L GH010353",
                        "FieldData 20200528110542 1L GH070073",
                        "FieldData 20200515101008 2R GH070120"
                    ],
    "save_dir": "/train/pickled_weed",
    "writer_dir": "/train/runs",
    "dataset_dir": "/weed_data",
    "sherpa_parameters": {
        "lr": [0.000005, 0.001],
        "weight_decay": [0.00001, 0.9],
        "momentum": [0.1, 0.9],
        "step_size": [1, 5],
        "gamma": [0.01, 0.5]
    }
}
```
- "variants" defines the networks to train and they have entries of their own which must contain at least "run_hpo" set to 0 or 1. This configures if Sherpa will search for hyper parameters or not. If set to 0 a good set of optimal parameters must be supplied for use during training. Last a general set of parameters are set. This concept is easily expandable to more settings.
- "optimal_hpo_settings", fill these in from the free standing settings file generated by the Sherpa run when you have something you want to keep.

## Metrics
- With the container active and the terminal ready and a model trained change dir to /code and run: `python3 test_model.py`


## Auto annotation
- Can be run like this: python3 auto_annotate.py -p /train/pickled_weed/pd_val.pkl -v resnet18 -t 0.5 for automatic annotation and uploading to cvat. For uploading of ground truth run python3 auto_annotate.py -p /train/pickled_weed/pd_val.pkl -g /auto_annotation/resnet18/2021_12_16_12_57_57. Rerun auto annotation on specific task: python3 auto_annotate.py -r "FieldData 20200515101008 2L GH070066" -t 0.7
- Update gt_tasks with auto annotations: python3 /code/auto_annotate.py -u True -m gt_val -d auto_annotation_resnet18_2021_12_16_12_57_57 where -m flag is pattern to match gt task with and -d is the match pattern for auto annotations. The code will match tasks by itself. The idea is to use a well trained detector with high confidence threshold to find missing annotations in the validation dataset and after manual validation push these back to the ground truth task.
- A special flag has been added for convenience, -w --whatever is connected to the function do_whatever and has been used to set a list of tasks to status completed for example. Modify this code to automate whatever.
- To upload annotations in xml format from a folder run: python3 /code/auto_annotate.py -x "/train/xml_annotations" as example path.


## Left to sort out
- how to do intelligent splits on train/validation data based on feature space mapping from t-sne. This is available on the rest-api on the analytics container. For now every 10th frame is used for validation. A train/val split is given where the validation dataset has been verified to has as few missing annotations as possible.



