import torch
import json
from param_parser import parameter_parser
import glob
from tqdm import tqdm
import utils



args = parameter_parser()
# print(args.training_graphs)

#读取目标路径文件名

def process_pair(path):
    data = json.load(open(path))
    return data

training_graphs = glob.glob(args.training_graphs + '*.json')

print(type(training_graphs))
print(training_graphs[0])
print(type(training_graphs[0]))

# data = utils.process_pair(training_graphs[0])
data = process_pair()



global_labels = set()
global_labels = global_labels.union(set(data['labels_1']))
# print(global_labels)

global_labels = sorted(global_labels)
# print(global_labels)
#
global_labels = {val:index for index,val in enumerate(global_labels)}
print(global_labels)


print(data['labels_1'])

features_1 = []
for n in data["labels_1"]:
    features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
print(features_1)


