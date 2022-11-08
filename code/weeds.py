import requests
from pymongo import MongoClient
import os

#class to fetch cvat annotations from mongoDB
class Weeds():
    def __init__(self, port=27017):
        self.db_client = MongoClient(host='0.0.0.0', port=port, 
                                        username=os.environ['MONGODB_USERNAME'], password=os.environ['MONGODB_PASSWORD'],
                                        connect=True, authSource="annotations")  

        self.db = self.db_client['annotations']
        self.db_collection_tasks = self.db.tasks
        self.db_collection_annotation_data = self.db.annotation_data
        self.db_collection_task_meta = self.db.meta


    def get_task_data(self, task_name):
        query = { "$regex": '^'+task_name }      
        return self.db_collection_tasks.find({'name': query})


    def get_all_db_tasks(self):
        return self.db_collection_tasks.find()
            

    def get_annotations_for_task(self, task_name):
        return self.db_collection_annotation_data.find({'task_name': task_name})


    def get_annotations_for_task_and_frame(self, task_name, frame_name):
        return self.db_collection_annotation_data.find({'task_name': task_name, 'img_path': frame_name})



    def get_object_classes_for_all_annotations(self):
        return self.db_collection_annotation_data.distinct('object_class')


    def get_object_classes_for_annotations_with_task_filter(self, filter):
        query = { "$regex": '^'+filter }
        return self.db_collection_annotation_data.distinct('object_class', {'task_name': query})


    def get_meta_for_task(self, task_name):
        return self.db_collection_task_meta.find({'task_name': task_name})

