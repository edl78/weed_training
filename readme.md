# Training repo for the OBDB project

For an overview of the Openweeds project, some background on this documentation and instructions on how to get the data, please see [obdb_docs](https://github.com/edl78/obdb_docs).

## Architecture
- The code depends on having a mongodb instance with all annotation data collected from CVAT, meaning weed_annotations must first be started (unless going for *fast-track to training*, see [obdb_docs](https://github.com/edl78/obdb_docs).
- Bayesian optimization performed via Optuna for hyperparameters.
- Can output a number of networks ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] or any combination of pytorch available networks. As a default the code is configured to train a resnet18 for the weed detection task.
- If fast track method is chosen, see [obdb_docs](https://github.com/edl78/obdb_docs), weed_training is stand alone and depends only on the image data and pickle files.


![](doc_img/architecture.png)



## How to run (and get data)
- Fill usernames and passwords in the env files `.env` and `env.list`, `.env` is used by docker-compose and `env.list` is sent as environment variables into the container.
### Accepting data license agreement
- You **MUST** accept the license agreement for the data by setting the variable `ACCEPT_ARTEFACTS_LICENSE` in the `env.list` to `YESPLEASE` to enable data download. No data will be downloaded before you accept the license for the data. See `LICENSE-DATA.md` for the license text.
### Needed to be run once at the start
- Shell scripts for building docker image: `sh build_training.sh`
- To download the images run full_hd (recommended and supported): `docker-compose -f docker-compose-download-full-hd.yml up` or for 4k versions: `docker-compose -f docker-compose-download-4k.yml up`
- To download the artefacts (annotations, pre-trained model and other necessary files) run: `docker-compose -f docker-compose-download-artefacts.yml up`
- To download camera calibration data (checker board images for each camera) run: `docker-compose -f docker-compose-download-calibration.yml up`
- To download tSNE data for the datasets run: `docker-compose -f docker-compose-download-tSNE.yml up`
- There is an optional dataset contributed by [WASP](https://wasp-sweden.org/) for segmentation. To download the WASP segmentations run: `docker-compose -f docker-compose-download-wasp.yml up` This dataset can be uploaded to cvat via `docker-compose -f docker-compose-upload-wasp-segmentation-data-cvat.yml up` and assumes the data folder with images named wasp is located in the fielddata folder.

### Training
All training depend on pickle files. These can be created (see Auto-annotations section below) or found in the `/train/pickled_weed/` folder. The path to the pickle files to be used can be set by changing the variables `TRAINING_PICKLE_PATH` and `VALIDATION_PICKLE_PATH` in the `.env` file. *Important to note that these paths are not host paths but rather paths as seen from the inside of the container using them.* Thus any pickle file must be found somewhere under the `train` folder as mounted by the docker-compose file.

To start training, the main docker-compose is used, run: `docker-compose up -d` (i.e. run without specific docker-compose file, run without -d for console output). To run interactive `run_training.sh` can be used to run interactive if so desired. Fill in missing usernames and passwords in the `run_training.sh` shell script before starting it.

The trained networks can then be found in the mapped folder `train` or `/train` in the container. A file with optimal training parameters is also located together with the network.

Bayesian hyper parameter search is implemented with the Optuna library. Use this by setting `run_hpo`: 1 in `code/settings_file_train_val.json` under respective network. 

To start the Optuna-dashboard:
- Run `docker ps` to list the active containers. Use the `CONTAINER ID` for the `training:v2 image` and
- Run `docker exec -it CONTAINER ID /bin/bash` which will give you a terminal to the container running the Optuna HPO.
- inside the container termainal run (choose your own prefered port number): `optuna-dashboard --port 8084 --net=host sqlite:///train/db.sqlite3:8087`
-   Watch the hyper parameter tuning on localhost:8084 or your port of choice and the training and validation losses for all runs on localhost:6006 after pointing your local tensorboard to code/runs/name_of_the_run



#### For fast-track
- Use pickle files for full_hd `pd_train_full_hd.pkl` and `pd_val_full_hd.pkl` found in `train/pickled_weed/`. These are used by default.
- Run `docker-compose up` add -d at later runs if no std output is desired.
- After training is completed a model is saved in `train/resnet18_model_weeds_pretrained.pth`, it can be tested with metrics with `docker-compose -f docker-compose-metrics.yml` which produces a json file with average precision per class in `train/AveragePrecision_YYYY_MM_DD_HH_MM_SS.json`
- To run Bayesian hyper parameter search, set `run_hpo` to `1` and `use_settings_file` to `1` in the json config file in `code/settings_file_train_val.json`.

#### Complete setup for training
- Upload annotations to CVAT by following the instructions in `To interact with CVAT` below.
- Load annotations into the MongoDB by following the instructions `Load data into MongoDB` below.
- Now you can make more annotations in CVAT by creating new tasks and annotate these. Set the tasks to complete status when finished with the annotations. 
- Follow the instructions for making a new dataset below, `Create pickle files`.
- After the above steps a new training run can be started with the newly added data.



#### Config example for training
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
    "annotations_list_4k": ["FieldData 20200520145736 1L GH020068",
                        "FieldData 20200515101008 3R GH070071",
                        "FieldData 20200603102414 1L GH010353",
                        "FieldData 20200528110542 1L GH070073",
                        "FieldData 20200515101008 2R GH070120"
                    ],
    "save_dir": "/train/pickled_weed",
    "writer_dir": "/train/runs",
    "dataset_dir": "/weed_data",
    "hpo_parameters": {
        "lr": [0.000005, 0.001],
        "weight_decay": [0.00001, 0.9],
        "momentum": [0.1, 0.9],
        "step_size": [1, 5],
        "gamma": [0.01, 0.5]
    }
}
```
#### Parameters for training 
These are set in the config json-file examplified above.
- `run_hpo`: 1 forces the training to do hyper parameter search. The found hyperparameters are written to `train/<YOUR_CHOSEN_MODEL>_settings.json`, e.g. `resnet18_settings.json`. Reset to `0` and set `use_settings_file` to `1` to start again with regular training using the found hyperparameters in the above file. 
- `full_hd` defines image size, set to one uses 1920x1080, otherwise 4k resolution can be utilized but full support might be lacking.
- `variants` defines the networks to train and they have entries of their own which must contain at least `run_hpo` set to 0 or 1. This configures if Sherpa will search for hyper parameters or not. If set to 0 a good set of optimal parameters must be supplied for use during training. Last a general set of parameters are set. This concept is easily expandable to more settings.
- `use_settings_file` if set to `1` uses parameters in `train/<YOUR_CHOSEN_MODEL>_settings.json` (which is created by the hyper parameter optimization run, see `run_hpo` above). Setting this to `1` disregards any hyperparameters in the config structure. Otherwise hyperparamters from `optimal_hpo_settings` in the config structure under respective variant will be used.
- `optimal_hpo_settings`, fill these in from the free standing settings file generated by the Sherpa run when you have something you want to keep.
- `fake_dataset_len` is used to define how much of the dataset should be used during hyper parameter optimizatin when we try to overfit to find powerful parameters. The parameter `fake_dataset_len` is also used as the optimization tries to overfit as agressively as possible on this small dataset. This overrides the dataset size in the dataloader during training.
- `default_class_map` is used when no MongoDB instance is used during fast track to training.
- `metrics_confidence_range` is an array to define confidence range for metrics.
- `metrics_iou_range` is an array to define iou range for metrics.
- `confidence_threshold_save_img` defines at what threshold to draw bounding boxes on the images.

### To populate CVAT with images and annotations (first time only)
To insert all the images and annoationas into CVAT, the following procedure must be followed:

- Start mongodb service in `weed_annotations` (will start several services but the mongodb is the central one for this workflow), see `readme.md` in that repository.
- Set similar `env.list` settings in `weed_training` as in `weed_annotations`.
- Populate the new empty MongoDB with relevant data, see *Alternative 1* below.
- Upload data to cvat (find docs in `obdb_docs` repo). 
  

#### Load data into MongoDB
There are two ways to load annotations into MongoDB.

*Alternative 1*: Load annotations into mongodb with mongoimport interface. Install the [MongoDB Database Tools](https://www.mongodb.com/docs/database-tools/installation/installation-linux/) by downloading from mongodb website and follow installation instructions. Find the mongodb json files in the artefacts folder downloaded from the OBDB site. To import data into MongoDB use (fill in your username, password and port):
- Known bugs in the bitnami/mongodb: must initialize with port 27017 and do not change root user name! Otherwise it will not work...
- https://www.mongodb.com/try/download/database-tools
- For the annotation data: `mongoimport --username= --password= --host=localhost --port= --collection=annotation_data --db=annotations annotation_data.json`
- For the meta data:
`mongoimport --username= --password= --host=localhost --port= --collection=meta --db=annotations meta.json`
- For the tasks data:
`mongoimport --username= --password= --host=localhost --port= --collection=tasks --db=annotations tasks.json`

*Alternative 2*: - Fetch all annotations via dashboard (available after starting weed_annotations) to MongoDB: press update annotations button in the dashboard running on localhost:8050
- Requires that you followed "To interact with CVAT" above and weed_annotations up and running.
- MongoDB contents can be viewed via localhost:8081, on the MongoExpress GUI.

#### Upload data to CVAT
- First try to upload the validation data to CVAT (it is smaller and thus a good place to start) by: `docker-compose -f docker-compose-upload-val-data-cvat.yml up`
- Upload training data by: `docker-compose -f docker-compose-upload-train-data-cvat.yml up`


### To interact with CVAT

- Set all tasks in cvat to status complete by running: `docker-compose -f docker-compose-set-all-cvat-tasks-to-complete.yml up` this is needed since the weed_annotations dashboard collects all annotations from tasks that are set in status complete and inserts them into MongoDB.


### Create pickle files
- Update the MongoDB by pressing `update annotations` on the `weed_annotations` dashboard if new annotations have been added in CVAT.
- Add any new CVAT tasks to the config file `code/settings_file_train_val.json`. 
- Add the training and validation frames in `/train/pickled_weed/*.npy`.
- Add names of pickle files in `.env` and `env.list` and add the name of the task list, `TASK_LIST_NAME=` to use in the `code/settings_file_train_val.json`. The pickle file will be created based on this task list and the list of validation and training frames in the `.npy` files.
- Set `full_hd` to 1 or 0 in the `code/settings_file_train_val.json` to set full_hd vs 4k. Create new pickle files for training by running `docker-compose-make-new-dataset.yml`. A good idea would be to name 4k pickle files `pd_train_4k.pkl` and `pd_val_4k.pkl` for example.


### Debugging
- In interactive mode, after starting the contrainer, go to `/code` and run `python3 torch_model_runner.py -f path_to_json_settings_file -t path_to_train_pkl_file -v path_to_val_pkl_file`


### Metrics
- Copy trained model to `/train/models`. Path to model can also be set in `docker-compose-metrics.yml`. Change other parameters as required.
- Run with: `docker-compose -f docker-compose-metrics.yml up`
- Output in `train/AveragePrecision_YY_MM_DD_HH_MM_SS.json`, where `YY_MM_DD_HH_MM_SS` is the date. 



## Auto annotation - life-cycle of annotations **TBD**
- Run with `docker-compose -f docker-compose-auto-annotate.yml up` 
- Auto annotation parameters: "-f": path to folder with images as "/weed_data/fielddata/tractor-32-cropped/20190524130212/1R/GH010033", "--ext": imgage file extension such as "png", "-t": confidence threshold as "0.7", "-i": iou threshold if run on a task with prior annotations to be able to complete missing annotations set to "0.7" for example, "--model_path": path to PyTorch model to use for auto annotation run like "/train/resnet18_model.pth", "--settings_file": is the settings file given on the format of "/code/settings_file_train_val.json".
- Setting file variables: "auto_annotation_class_map": used to map network output to object classes.
- After Auto annotation run is finished and the data and annotations are uploaded to cvat, inspect annotations and correct if needed. Set the task to completed state and go to the weed_annotations dashboard. Update annotations to import them into MongoDB. Then decide on a training/validation split and update the files train_frames_full_hd.npy and val_frames_full_hd.npy that define the frames to include in the dataset.
- Make a new dataset in two ways: By including the flag -m "True" in the weed_training docker-compose.yml. By running this docker-compose a new training will also be started. Or start the separate compose file: `docker-compose -f docker-compose-make-new-dataset.yml up`


# Academic citation

Please see the [main webpage](https://openweeds.linkoping-ri.se/howtocite.html) for information on how to cite.

# Issues?

Please use the github issues for this repo to report any issues on it.