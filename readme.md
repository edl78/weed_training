# Training repo for obdb

## Architecture
- Depends on having a mongodb instance with all annotation data collected from CVAT, meaning weed_annotations must first be started.
- Bayesisan optimization via Sherpa for hyperparameters.
- Can output a number of networks ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] or any combination of pytorch available networks. Configured to train a resnet18 for the weed detection task.
- If fast track method is chosen, weed_training is stand alone and depends only on the image data and pickle files.


![](doc_img/architecture.png)



## How to run
- Shell scripts for building: `sh build_training.sh`
- To download the images run full_hd (recommended and supported) or 4k versions of `docker-compose-download-full-hd.yml`
- Upload data to cvat once the cvat service is up (find docs in obdb_docs repo) with: `docker-compose-upload-train-data-cvat.yml` and `docker-compose-upload-val_data-cvat.yml`
- Set all tasks in cvat to status complete by running: `docker-compose-set-all-cvat-tasks-to-complete.yml` this is needed since the weed_annotations dashboard collects all annotations from tasks that are set in status complete and inserts them into MongoDB.
- Fill usernames and passwords in the env files .env and env.list, .env is used by docker-compose and env.list is sent as environment variables into the container.
- Run with docker-compose: `docker-compose up -d` run without -d for console output.
- run_training.sh can be used to run interactive if so desired. Fill in missing usernames and passwords in the shell script.
- In interactive mode, after starting the contrainer, go to /code and run `python3 torch_model_runner.py -f path_to_json_settings_file -t path_to_train_pkl_file -v path_to_val_pkl_file`
- The trained networks can then be found in the mapped folder train or /train in the container. A file with optimal training parameters is also located together with the network.
- Bayeisan hyper parameter search is implemented with the Sherpa library. Use this by setting "run_hpo": 1 in code/settings_file_gt_train_val.json under respective network. The parameter "fake_dataset_len" is also used as the optimization tries to overfit as agressively as possible on this small dataset. This overrides the dataset size in the dataloader during training.
- Watch the hyper parameter tuning on localhost:8880 and the training and validation losses for all runs on localhost:6006 after pointing your local tensorboard to code/runs/name_of_the_run
- The json configuration file looks like this: 
```json
{
    "full_hd": 1,
    "variants": ["resnet18","resnet34"],
    "resnet18": {
        "run_hpo": 0,
        "use_settings_file": 1,
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
- "full_hd" defines image size, set to one uses 1920x1080, otherwise 4k resolution can be utilized but full support might be lacking.
- "variants" defines the networks to train and they have entries of their own which must contain at least "run_hpo" set to 0 or 1. This configures if Sherpa will search for hyper parameters or not. If set to 0 a good set of optimal parameters must be supplied for use during training. Last a general set of parameters are set. This concept is easily expandable to more settings.
- "use_settings_file" if set to 1 points to /train/"variant"_settings_file.json which is created by the hyper paramter optimization run. Otherwise hyperparamters from "optimal_hpo_settings" under respective variant will be used.
- "optimal_hpo_settings", fill these in from the free standing settings file generated by the Sherpa run when you have something you want to keep.
- "fake_dataset_len" is used to define how much of the dataset should be used during hyper parameter optimizatin when we try to overfit to find powerful parameters.
- "default_class_map" is used when no MongoDB instance is used during fast track to training.
- "metrics_confidence_range" is an array to define confidence range for metrics.
- "metrics_iou_range" is an array to define iou range for metrics.
- "confidence_threshold_save_img" defines at what threshold to draw bounding boxes on the images.


## Metrics
- Run this with: `docker-compose -f docker-compose-metrics.yml up`
- Output in train/class_metrics.json file which is used to produce graphs default put under train/result_figures/ with a number of svg files. Replace up with down to remove the container when done in the docker-compose command.



## Auto annotation
- Run with `docker-compose -f docker-compose-auto-annotate.yml up` 
- Auto annotation parameters: "-f": path to folder with images as "/weed_data/fielddata/tractor-32-cropped/20190524130212/1R/GH010033", "--ext": imgage file extension such as "png", "-t": confidence threshold as "0.7", "-i": iou threshold if run on a task with prior annotations to be able to complete missing annotations set to "0.7" for example, "--model_path": path to PyTorch model to use for auto annotation run like "/train/resnet18_model.pth", "--settings_file": is the settings file given on the format of "/code/settings_file_gt_train_val.json".
- Setting file variables: "auto_annotation_class_map": used to map network output to object classes.
- After Auto annotation run is finished and the data and annotations are uploaded to cvat, inspect annotations and correct if needed. Set the task to completed state and go to the weed_annotations dashboard. Update annotations to import them into MongoDB. Then decide on a training/validation split and update the files train_frames_full_hd.npy and val_frames_full_hd.npy that define the frames to include in the dataset.
- Make a new dataset in two ways: By including the flag -m "True" in the weed_training docker-compose.yml. By running this docker-compose a new training will also be started. Or start the separate compose file: `docker-compose -f docker-compose-make-new-dataset.yml up`, replace up by down when done in the docker-compose command.





