from auto_annotate import calculate_iou
import json
from weeds import Weeds
import matplotlib.pyplot as plt
import numpy as np
import os

class DetectionMetrics():
    def __init__(self, class_map=None, confidence_thresholds=[0.7], iou_thresholds=[0.5]):
        self.num_classes = len(class_map)
        self.class_map = class_map
        self.confidence_thresholds = confidence_thresholds
        self.iou_thresholds = iou_thresholds
        self.metrics = self.get_base_metrics()            
        self.class_metrics = dict()
        for object_class in self.class_map:
            self.class_metrics[object_class] = self.get_base_metrics()


    def get_base_metrics(self):
        metrics = {
            #format = 2d dict, for every confidence_threshold there is a corresponding iou_threshold dict
            #result dict = {"confidence_threshold": float, "iou_threshold": float, "metric" }
            'TP' : self.get_2d_threshold_grid_dict(), #detections with gt
            'FP' : self.get_2d_threshold_grid_dict(), #detection without gt
            'FN' : self.get_2d_threshold_grid_dict(),  #gt without detection
            'precision': self.get_2d_threshold_grid_dict(),
            'recall': self.get_2d_threshold_grid_dict(),
            'f1': self.get_2d_threshold_grid_dict(),
        }
        return metrics

    
    def get_2d_threshold_grid_dict(self):
        confidence_thresholds = dict()
        for val_c in self.confidence_thresholds:
            iou_thresholds = dict()
            for val_i in self.iou_thresholds:
                iou_thresholds[str(val_i)] = 0
            confidence_thresholds[str(val_c)] = iou_thresholds 
        
        return confidence_thresholds


    def get_frame_metrics(self):
        metrics = {
            #format = [[frame0 result dict][frame 1 result dict]]
            #result dict = {"confidence_threshold": float, "iou_threshold": float, "metric" }
            'TP' : 0, #detections with gt
            'FP' : 0, #detection without gt
            'FN' : 0  #gt without detection
        }
        return metrics


    def update(self, predictions=None, gt=None):
        #add target and prediction result to metrics
        #format predictions = {"boxes": [], "labels": [], "scores": []}
        #format gt = {"boxes": [], "labels": []}
        
        for confidence_threshold in self.confidence_thresholds:
            filtered_scores = [score for score in predictions['scores'] if score >= confidence_threshold]
            for iou_threshold in self.iou_thresholds:
                class_metrics_current_frame = dict()
                for object_class in self.class_map:
                    class_metrics_current_frame[object_class] = self.get_frame_metrics()
                
                class_metrics_current_frame['confidence_threshold'] = confidence_threshold
                class_metrics_current_frame['iou_threshold'] = iou_threshold                
                for i in range(len(filtered_scores)):
                    #TP, FP
                    current_label =  predictions['labels'][i]
                    #assume FP, decrease counter if not correct
                    class_metrics_current_frame[self.class_map[current_label]]['FP'] += 1
                    for j in range(len(gt['labels'])):
                        if(current_label == gt['labels'][j]):
                            iou = calculate_iou(predictions['boxes'][i], gt['boxes'][j])
                            if(iou >= iou_threshold):
                                class_metrics_current_frame[self.class_map[current_label]]['TP'] += 1                             
                                class_metrics_current_frame[self.class_map[current_label]]['FP'] -= 1
                            
                                            
                #check FN, loop through gt for missing match
                for k in range(len(gt['labels'])):
                    #assume FN
                    current_label = gt['labels'][k]
                    class_metrics_current_frame[self.class_map[current_label]]['FN'] += 1
                    for r in range(len(filtered_scores)):
                        iou = calculate_iou(predictions['boxes'][r], gt['boxes'][k])
                        if(iou >= iou_threshold and current_label == predictions['labels'][r]):
                            class_metrics_current_frame[self.class_map[current_label]]['FN'] -= 1
                            break
                
                #add frame metrics to class_metrics
                for object_class in self.class_map:        
                    conf_t = class_metrics_current_frame['confidence_threshold']
                    iou_t = class_metrics_current_frame['iou_threshold']            
                    self.class_metrics[object_class]['TP'][str(conf_t)][str(iou_t)] += class_metrics_current_frame[object_class]['TP']
                    self.class_metrics[object_class]['FP'][str(conf_t)][str(iou_t)] += class_metrics_current_frame[object_class]['FP']
                    self.class_metrics[object_class]['FN'][str(conf_t)][str(iou_t)] += class_metrics_current_frame[object_class]['FN']
            


    def calc_metrics(self, metrics_save_path=None):
        #precision = TP/(TP+FP) = TP/all detections
        #recall = TP/(TP+FN) = TP/all gt
        #F1 = 2*Precison*Recall/(Precision + Recall)
        #save one complete dict() as json
        
        #metrics per class:
        for object_class in self.class_map:
            for conf_t in self.confidence_thresholds:
                for iou_t in self.iou_thresholds:
                    print('metrics for ' +  object_class + ' conf_t: ' + str(conf_t) + \
                         ' iou_t: ' + str(iou_t) + ':')
                    tp = self.class_metrics[object_class]['TP'][str(conf_t)][str(iou_t)]
                    fp = self.class_metrics[object_class]['FP'][str(conf_t)][str(iou_t)]
                    fn = self.class_metrics[object_class]['FN'][str(conf_t)][str(iou_t)]
                    tp_fp = tp + fp
                    tp_fn = tp + fn
                    if(tp_fp > 0):
                        pr = tp/(tp + fp)
                    else:
                        pr = 0
                    if(tp_fn > 0):
                        rc = tp/(tp + fn)
                    else:
                        rc = 0
                    
                    if(pr + rc > 0):
                        f1 = 2*pr*rc/(pr + rc)
                    else:
                        f1 = 0
                    print('precision: ' + str(pr))
                    print('recall: ' + str(rc))
                    print('f1: ' + str(f1))
                    print('total tp: ' + str(tp))
                    print('total fp: ' + str(fp))
                    print('total fn: ' + str(fn))
                    
                    self.class_metrics[object_class]['precision'][str(conf_t)][str(iou_t)] = pr
                    self.class_metrics[object_class]['recall'][str(conf_t)][str(iou_t)] = rc
                    self.class_metrics[object_class]['f1'][str(conf_t)][str(iou_t)] = f1
            
    
        with open(metrics_save_path, 'w') as fout:
            json.dump(self.class_metrics , fout)

    

    def make_graphs(self, metrics_json_path=None, figure_save_folder=None):
        metrics = None
        with open(metrics_json_path) as json_file:            
                metrics = json.load(json_file)
        
        os.makedirs(figure_save_folder, exist_ok=True)

        #For every class, make graphs (precision, recall, f1) of each confidence threshold with varying iou threshold
        #print in one class graph        
        
        for object_class in self.class_map:
            precision = metrics[object_class]['precision']
            recall = metrics[object_class]['recall']
            f1 = metrics[object_class]['f1']

            graph_dict = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            for metric in graph_dict.keys():
                plt.clf()
                plt.plot()    
                plt.xlabel('iou_threshold')
                plt.ylabel(metric)
                plt.title(object_class + ' metrics for varying confidence and iou thresholds')                
                plt.grid(True)
                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 
                            'tab:gray', 'tab:olive', 'tab:cyan', 'darkblue', 'lime', 'darkred', 'yellow']
                
                i=0
                ious = []
                for key in graph_dict[metric].keys():                                        
                    vals = []
                    
                    for iou in graph_dict[metric][key]: 
                        ious.append(np.float(graph_dict[metric][key][iou]))
                        vals.append(np.float(graph_dict[metric][key][iou]))
                                                            
                    plt.plot(vals, color=colors[i], label=str(key) + ' conf')
                    i += 1
                y_auto = np.max(ious)
                #handle zero vals
                y_auto_lim = np.max([y_auto, 0.0001])
                plt.axis([0.5, 0.9, 0, y_auto_lim])
                plt.legend()
                plt.savefig(figure_save_folder + '/' + object_class + '_' +  metric + '.svg', format='svg')

        print('graphs done, find them in ' + figure_save_folder)