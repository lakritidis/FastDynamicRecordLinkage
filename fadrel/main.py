import pandas as pd
import paths

from sklearn.model_selection import KFold

from FADREL_prepare import FADRELPreparationPhase
from FADREL_match import FADRELMatchingPhase
from Tools import set_random_states, get_random_states, reset_random_states


random_state = 42
max_embeddings_length = 32
epochs = 5
batch_size = 16
evaluate_classifier = False
positive_title_pairs = 3
negative_label_pairs = 7
negative_title_pairs = 0


if __name__ == '__main__':
    # Initialize the random number generators
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    # Original dataset
    dataset_name = 'pricerunner_mobile_clean'
    df = pd.read_csv(
        paths.datasets_path + dataset_name + '.csv',
        names=['Product ID', 'Product Title', 'Vendor ID', 'Cluster ID', 'Cluster Label', 'Category ID',
               'Category Label'])

    paths.create_output_dirs(dataset_name + '/')

    # Apply Cross Validation (CV)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    label_column = 'Cluster Label'
    label_id_column = 'Cluster ID'
    title_column = 'Product Title'

    fold = 0
    eval_list = []
    eval_path = dataset_name + '/' + paths.models_path + paths.model_eval_file

    # For each CV fold
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        fold += 1

        print("\n===\n=== Processing fold " + str(fold))

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        method_name = "neg" + str(negative_label_pairs) + "_pos" + str(positive_title_pairs) + "_fold" + str(fold)

        fold_paths = {
            'bert_path': paths.output_path + dataset_name + '/' + paths.models_path + paths.seq_class_file + '_fold' + str(fold),
            'train_path': paths.output_path + dataset_name + '/' + paths.aux_path + paths.train_pairs_file + '_fold' + str(fold) + '.csv',
            'query_path': paths.output_path + dataset_name + '/' + paths.aux_path + paths.query_pairs_file + '_fold' + str(fold) + '.csv',
            'lab_index_path': paths.output_path + dataset_name + '/' + paths.aux_path + paths.lab_index_file + '_fold' + str(fold),
            'cl_data_path': paths.output_path + dataset_name + '/' + paths.aux_path + paths.cluster_data_file + '_fold' + str(fold)
        }

        # Offline pipeline
        offline_pipe = FADRELPreparationPhase(dataset_name=dataset_name, entity_id_col=label_id_column, entity_label_col=label_column,
                                              title_col=title_column, paths=fold_paths,
                                              max_emb_len=max_embeddings_length, epochs=epochs, batch_size=batch_size,
                                              num_neg_pairs_labels=negative_label_pairs, num_pos_pairs_titles=positive_title_pairs,
                                              num_neg_pairs_titles=negative_title_pairs,
                                              random_state=random_state)
        offline_pipe.run(train_df)

        # Online pipeline
        online_pipe = FADRELMatchingPhase(dataset_name=dataset_name, method_name=method_name, label_id_col=label_id_column,
                                          label_str_col=label_column, title_col=title_column, paths=fold_paths, epochs=epochs,
                                          max_emb_len=max_embeddings_length, batch_size=batch_size, random_state=random_state)
        online_pipe.run(test_df)
