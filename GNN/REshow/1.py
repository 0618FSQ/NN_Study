import json
import torch
import glob
from GNN.similarity.param_parser import parameter_parser
from tqdm import tqdm


def process_pair(path):
    data = json.load(open(path))
    return data



args = parameter_parser()


class SimGNNTrainer(object):
    def __init__(self):
        self.args = parameter_parser()
        self.initialization_labels()

    def initialization_labels(self):
        self.training_graphs = glob.glob(self.args.training_graphs + '*.json')
        self.testing_graphs = glob.glob(self.args.training_graphs + '*.json')

        self.graphs_pairs = self.training_graphs + self.testing_graphs

        self.global_labels = set()

        for pair in tqdm(self.graphs_pairs):
            data = process_pair(pair)
            self.global_labels = self.global_labels.union(set(data['labels_1']))
            self.global_labels = self.global_labels.union(set(data['labels_2']))

        self.global_labels = sorted(self.global_labels)
        self.numbers_of_labels = len(self.global_labels)

    def To_Tensor(self,data):
        new_data = {}
        for pair in data['graph_1']:
            new_data['graph_1'] = pair + [(y,x) for x,y in data['graph_1']]
        pass


    def create_batches(self):
        batch_size = self.args.batch_size
        batches = []

        for i in tqdm(range(0,len(self.training_graphs),batch_size)):
            batches.append(self.training_graphs[i:i+batch_size])
        return batches

    # def batch_process(self):
    #     batches = self.create_batches()
    #     pass



path = SimGNNTrainer().training_graphs[0]
data = json.load(open(path))
print(data.keys())










