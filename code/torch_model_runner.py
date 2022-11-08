import sys
import json
import argparse
import os
from weed_data_class_od import WeedDataOD
from create_weed_dataset_od import PickledWeedOD
from model_trainer import ModelTrainer
#from model_trainer_overfit import ModelTrainer
from create_weed_dataset_od import PickledWeedOD
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description='training file for weeds, Sherpa is used for Bayesian HPO')
    parser.add_argument('-f', '--settings_file', type=str, default='/code/settings_file_gt_train_val.json', help='settings file for all settings regarding the networks to train', required=False)
    args = parser.parse_args()    

    with open(args.settings_file) as json_file:            
            settings = json.load(json_file)

    #setup pandas dataset    
    save_dir = settings['save_dir']
    pickledWeed = PickledWeedOD(task_name_list=settings['annotations_list_gt'],
                                save_dir=save_dir,
                                dataset_dir=settings['dataset_dir'],
                                mongo_port=os.environ['MONGODB_PORT'])
    
    #pickledWeed.make_pandas_dataset(pickle_name='pd_val_first_year.pkl')
    full_hd = settings['full_hd']
    pickledWeed.make_pandas_dataset_with_pre_split(full_hd=full_hd)
    #print('done, pickled pandas frame found at: ' + save_dir)    
    #pickledWeed.make_simple_split(save_dir)
    #pickledWeed.get_local_files()
    class_map = pickledWeed.get_class_map(filter='gt_')
    #class_map = pickledWeed.get_class_map(filter='FieldData_20190604')
    print(class_map)

    dataset = '/train/pickled_weed/pd_train.pkl'
    dataset_test = '/train/pickled_weed/pd_val.pkl'

    for variant in settings['variants']:
        trainer = ModelTrainer(dataset_path=dataset, 
                                dataset_test_path=dataset_test,
                                settings=settings, variant=variant,
                                class_map=class_map)
                        
        if(settings[variant]['run_hpo'] == True):
            #improve overfit on small batch, fix interface so we can run this
            #without any manual fixes (batchsize and len of dataset)
            trainer.sherpa_hpo()

        #point out newly created settings file for use during first run
        #after parameter search
        trainer.train_model(settings_file='/train/resnet18_settings.json')        
        #trainer.train_model()

if __name__ == "__main__":
    main()
