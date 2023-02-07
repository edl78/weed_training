import sys
import json
import argparse
import os
#from weed_data_class_od import WeedDataOD
from create_weed_dataset_od import PickledWeedOD
from model_trainer import ModelTrainer
from model_trainer_overfit import ModelTrainer_overfit
#from create_weed_dataset_od import PickledWeedOD
#from torchvision import transforms
import pandas as pd



def get_num_examples_per_class_in_dataset(dataset):
    print('statistics for dataset: ' + dataset)
    d = pd.read_pickle(dataset)
    d_all_labels = d.explode('labels')
    class_list = d_all_labels['labels'].unique()
    for c in class_list:
        df_c = d_all_labels[d_all_labels['labels'] == c]
        num_in_class = len(df_c.index)
        print(c + ' : ' + str(num_in_class))


def main():
    parser = argparse.ArgumentParser(description='training file for weeds, Sherpa is used for Bayesian HPO')
    parser.add_argument('-f', '--settings_file', type=str, default='/code/settings_file_gt_train_val.json', help='settings file for all settings regarding the networks to train', required=False)
    parser.add_argument('-t', '--dataset_training', type=str, default='/train/pickled_weed/pd_train_full_hd.pkl', help='training pickle file', required=False)
    parser.add_argument('-v', '--dataset_validation', type=str, default='/train/pickled_weed/pd_val_full_hd.pkl', help='validation pickle file', required=False)
    parser.add_argument('-m', '--make_new_dataset', type=bool, default=False, help='set to True to make a new pickle dataset', required=False)
    parser.add_argument('--no_training', type=bool, default=False, help='set to True to skip training when only a new dataset is required', required=False)
    parser.add_argument('-l', '--list_of_tasks', type=str, default='annotations_list_gt', help='settings file list name to build pickle file from', required=False)
    args = parser.parse_args()    

    with open(args.settings_file) as json_file:            
            settings = json.load(json_file)

    #setup pandas dataset    
    save_dir = settings['save_dir']
    class_map = settings['default_class_map']
    try:
        pickledWeed = PickledWeedOD(task_name_list=settings[args.list_of_tasks],
                                    save_dir=save_dir,
                                    dataset_dir=settings['dataset_dir'],
                                    mongo_port=int(os.environ['MONGODB_PORT']))

    except:
        print('error setting up MongoDB connection, this is ok if fasttrack to training is choosen.')
        if(args.make_new_dataset):
            print('must have a connection with MongoDB to make a new dataset, aborting')
            return

    if(args.make_new_dataset):
        print('make new dataset', flush=True)
        pickledWeed.make_pandas_dataset_with_pre_split(full_hd=settings['full_hd'], excluded_frames=settings['excluded_frames'],
                                                        train_pickle_path=args.dataset_training, val_pickle_path=args.dataset_validation)
        print('done, pickled pandas frame found at: ' + args.dataset_training + 'and: ' + args.dataset_validation)    
    
        if(args.no_training):
            print('skip training, just make new dataset')
            return
        
    class_map = settings['default_class_map']
    
    dataset = args.dataset_training
    dataset_test = args.dataset_validation

    #print stats of datasets
    get_num_examples_per_class_in_dataset(dataset)
    get_num_examples_per_class_in_dataset(dataset_test)

    for variant in settings['variants']:
        if(settings[variant]['run_hpo'] == True):
            trainer = ModelTrainer_overfit(dataset_path=dataset, 
                                dataset_test_path=dataset_test,
                                settings=settings, variant=variant,
                                class_map=class_map,
                                fake_dataset_len=settings['fake_dataset_len'])
            
            trainer.sherpa_hpo()
            print('HPO done, run training again without the run_hpo setting to train with new parameters, \
             also make sure to activate use_settings_file to use the newly found parameters otherwise manually \
             insert these in the training settings file')
        else:
            trainer = ModelTrainer(dataset_path=dataset, 
                                    dataset_test_path=dataset_test,
                                    settings=settings, variant=variant,
                                    class_map=class_map)
        
            if(settings[variant]["use_settings_file"]):
                trainer.train_model(settings_file='/train/' + variant + '_settings.json')
            else:
                trainer.train_model()

if __name__ == "__main__":
    main()
