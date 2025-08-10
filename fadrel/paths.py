import os

# Editable directory variables
# The input dataset location in the local filesystem
# datasets_path = '/media/leo/7CE54B377BB9B18B/datasets/EntityResolution/ProductMatching/'
datasets_path = 'D:/datasets/EntityResolution/ProductMatching/'

# The project location in the local filesystem
# project_path = '/media/leo/7CE54B377BB9B18B/dev/Python/FastDynamicRecordLinkage/'
project_path = 'D:/dev/Python/FastDynamicRecordLinkage/'

#
# Project Directory Structure - Do not modify beyond this line
output_path = project_path + 'runs/'
results_path = project_path + 'results/'

aux_path = 'intermediate/'
train_pairs_file = 'TrainPairs'
query_pairs_file = 'QueryPairs'
lab_index_file = 'LabelInvertedIndex'
tit_index_file = 'TitleInvertedIndex'
cluster_data_file = 'ClusterData'

models_path = 'models/'
seq_class_file = 'SequenceClassifier'
model_eval_file = 'model_eval.csv'

def create_output_dirs(dataset_name):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    if not os.path.isdir(output_path + dataset_name):
        os.mkdir(output_path + dataset_name)

    if not os.path.isdir(output_path + dataset_name + aux_path):
        os.mkdir(output_path + dataset_name + aux_path)

    if not os.path.isdir(output_path + dataset_name + models_path):
        os.mkdir(output_path + dataset_name + models_path)

