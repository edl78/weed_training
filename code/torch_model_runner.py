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


def main():
    parser = argparse.ArgumentParser(description='training file for weeds, Sherpa is used for Bayesian HPO')
    parser.add_argument('-f', '--settings_file', type=str, default='/code/settings_file_gt_train_val.json', help='settings file for all settings regarding the networks to train', required=False)
    parser.add_argument('-t', '--dataset_training', type=str, default='/train/pickled_weed/pd_train_full_hd.pkl', help='training pickle file', required=False)
    parser.add_argument('-v', '--dataset_validation', type=str, default='/train/pickled_weed/pd_val_full_hd.pkl', help='validation pickle file', required=False)
    parser.add_argument('-m', '--make_new_dataset', type=bool, default=False, help='set to True to make a new pickle dataset', required=False)
    args = parser.parse_args()    

    with open(args.settings_file) as json_file:            
            settings = json.load(json_file)

    #argparse hack
    make_new_dataset = args.make_new_dataset == 'True'

    #setup pandas dataset    
    save_dir = settings['save_dir']
    class_map = settings['default_class_map']
    try:
        pickledWeed = PickledWeedOD(task_name_list=settings['annotations_list_gt'],
                                    save_dir=save_dir,
                                    dataset_dir=settings['dataset_dir'],
                                    mongo_port=int(os.environ['MONGODB_PORT']))
        
        if(make_new_dataset):                
            pickledWeed.make_pandas_dataset_with_pre_split(full_hd=settings['full_hd'])
            print('done, pickled pandas frame found at: ' + save_dir)    
        
        #pickledWeed.get_local_files()
        class_map = pickledWeed.get_class_map(filter='gt_')    
        print(class_map)
    
    
    except:
        print('error setting up MongoDB connection, this is ok if fasttrack to training is choosen.')
        
    

    dataset = args.dataset_training
    dataset_test = args.dataset_validation

    for variant in settings['variants']:
        if(settings[variant]['run_hpo'] == True):
            trainer = ModelTrainer_overfit(dataset_path=dataset, 
                                dataset_test_path=dataset_test,
                                settings=settings, variant=variant,
                                class_map=class_map,
                                fake_dataset_len=settings['fake_dataset_len'])
            
            trainer.sherpa_hpo()
            #use newly created settings file after HPO
            trainer.train_model(settings_file='/train/' + variant + '_settings.json')

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
