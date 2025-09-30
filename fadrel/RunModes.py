from typing import Any, List
import pandas as pd

import paths
from sklearn.model_selection import KFold

from FADREL_prepare import FADRELPreparationPhase
from FADREL_match import FADRELMatchingPhase
from Tools import set_random_states, get_random_states, reset_random_states

from torch.utils.data import DataLoader

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import ContrastiveLoss, CosineSimilarityLoss, SiameseDistanceMetric

random_state = 42
max_embeddings_length = 64
epochs = 1
batch_size = 16
positive_pairs = 3
negative_pairs = 3


def set_fold_paths(dataset_name, fold):
    fold_paths = {
        'bert_path': paths.output_path + dataset_name + '/' + paths.models_path + paths.seq_class_file +
                     '_fold' + str(fold),
        'vectorizer_path': paths.output_path + dataset_name + '/' + paths.models_path + paths.vectorizer_file +
                           '_fold' + str(fold),
        'pairs_file': paths.output_path + dataset_name + '/' + paths.aux_path + paths.train_pairs_file +
                      '_fold' + str(fold) + '.csv',
        'triplets_file': paths.output_path + dataset_name + '/' + paths.aux_path + paths.train_triplets_file +
                      '_fold' + str(fold) + '.csv',
        'query_path': paths.output_path + dataset_name + '/' + paths.aux_path + paths.query_pairs_file +
                      '_fold' + str(fold) + '.csv',
        'inverted_index_datafile': paths.output_path + dataset_name + '/' + paths.aux_path + paths.inv_index_file +
                          '_fold' + str(fold),
        'records_datafile': paths.output_path + dataset_name + '/' + paths.aux_path + paths.records_data_file +
                        '_fold' + str(fold),
        'entities_datafile': paths.output_path + dataset_name + '/' + paths.aux_path + paths.entities_data_file +
                             '_fold' + str(fold)
    }
    return fold_paths


def mode_cross_validate():
    # Initialize the random number generators
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    # Original dataset
    dataset_name = 'irons_steamers'
    df = pd.read_csv(
        paths.datasets_path + dataset_name + '.csv',
        names=['Product ID', 'Product Title', 'Vendor ID', 'Cluster ID', 'Cluster Label', 'Category ID',
               'Category Label'])

    paths.create_output_dirs(dataset_name + '/')

    entity_id_col = 'Cluster ID'
    record_title_col = 'Product Title'

    # Apply Cross Validation (CV)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    fold = 0
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        fold += 1

        print("\n===\n=== Processing fold " + str(fold))

        train_df = df.iloc[train_idx].drop_duplicates(record_title_col).copy()
        test_df = df.iloc[test_idx].copy()
        method_name = "neg" + str(negative_pairs) + "_pos" + str(positive_pairs) + "_fold" + str(fold)

        fold_paths = set_fold_paths(dataset_name, fold)

        # Offline pipeline - Preparation Phase
        offline_pipe = FADRELPreparationPhase(dataset_name=dataset_name, entity_id_col=entity_id_col,
                                              record_title_col=record_title_col, paths=fold_paths,
                                              max_emb_len=max_embeddings_length, epochs=epochs, batch_size=batch_size,
                                              num_neg_pairs=negative_pairs, num_pos_pairs=positive_pairs,
                                              finetune_sbert=True, random_state=random_state)
        offline_pipe.run(train_df)

        # Online pipeline - Matching Phase
        online_pipe = FADRELMatchingPhase(dataset_name=dataset_name, method_name=method_name,
                                          entity_id_col=entity_id_col, record_title_col=record_title_col,
                                          paths=fold_paths, epochs=epochs, max_emb_len=max_embeddings_length,
                                          batch_size=batch_size, random_state=random_state)
        online_pipe.run(test_df)


def mode_sbert_contrastive():
    """
    Using SentenceTransformers for contrastive learning is straightforward and highly recommended—it abstracts away
    much of the complexity and supports contrastive losses like:
      - MultipleNegativesRankingLoss (SimCSE, retrieval-style training)
      - TripletLoss
      - ContrastiveLoss (custom)
      - CosineSimilarityLoss (STS-style)

    TL;DR: If you're using SentenceTransformers, the best way to do contrastive learning is with
    `MultipleNegativesRankingLoss` (uses positive pairs and in-batch negatives — no need to label negatives)
    """

    # Initialize the random number generators
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    # Original dataset
    dataset_name = 'irons_steamers'
    df = pd.read_csv(
        paths.datasets_path + dataset_name + '.csv',
        names=['Product ID', 'Product Title', 'Vendor ID', 'Cluster ID', 'Cluster Label', 'Category ID',
               'Category Label'])

    label_id_column = 'Cluster ID'
    title_column = 'Product Title'

    paths.create_output_dirs(dataset_name + '/')

    # Apply Cross Validation (CV)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    fold = 0
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        fold += 1

        print("\n===\n=== Processing fold " + str(fold))

        train_df = df.iloc[train_idx].drop_duplicates(title_column).copy()
        test_df = df.iloc[test_idx].copy()

        fold_paths = set_fold_paths(dataset_name, fold)

        # Offline pipeline
        offline_pipe = FADRELPreparationPhase(dataset_name=dataset_name, entity_id_col=label_id_column,
                                              record_title_col=title_column, paths=fold_paths,
                                              max_emb_len=max_embeddings_length, epochs=epochs, batch_size=batch_size,
                                              num_neg_pairs=negative_pairs, num_pos_pairs=positive_pairs,
                                              random_state=random_state)

        offline_pipe.initialize(train_df=train_df)
        training_pairs_df = offline_pipe.create_training_pairs()

        train_examples: List[Any] = [None] * training_pairs_df.shape[0]
        for n_row, row in enumerate(training_pairs_df.itertuples()):
            train_examples[n_row] = InputExample(texts=[row[1], row[2]], label=1)
            print(train_examples[n_row])

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_loss = ContrastiveLoss(model=model, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin=0.5)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=2,
            warmup_steps=100,
            show_progress_bar=True
        )

def mode_cluster_labeling():
    dataset_path = 'D:/datasets/EntityResolution/ProductMatching/wdc/'
    dataset_name = 'wdc_computers_train_xlarge'

    df = pd.read_csv(dataset_path + dataset_name + '_pairs.csv', names=['t1', 't2', 'y'], sep=',')
    train_examples : List[Any] = [None] * df.shape[0]
    for n_row, row in enumerate(df.itertuples()):
        train_examples[n_row] = InputExample(texts=[row[1], row[2]], label=row[0])

    # Load model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = CosineSimilarityLoss(sbert_model)

    # Fine-tune SentenceBERT model
    # sbert_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)

    df = pd.read_csv(dataset_path + dataset_name + '_clusters.csv', names=['cluster_id', 'cluster_members'], sep=';')

    # Initialize a KeyBERT model for assigning cluster labels
    kw_model = KeyBERT(model=sbert_model)

    ctr = 0
    stop_words = ['prijzen', 'wholesale', 'uk', 'price', 'prices', 'new', 'free', 'pcpartpicker', 'tweakers',
                  'hosting', 'powerplanetonline', 'ocuk']
    entities = {}
    for index, row in df.iterrows():
        cluster_id = row['cluster_id']
        cluster_members = row['cluster_members']
        member_titles = cluster_members.split(',')

        joined_docs = " ".join(member_titles)
        print("Member Titles:", member_titles)
        cand_tit = kw_model.extract_keywords(joined_docs, keyphrase_ngram_range=(3, 5), stop_words=stop_words, top_n=3)

        print("Candidate titles:", cand_tit)
        title = []
        for ct in cand_tit:
            words = ct[0].split(" ")
            for w in words:
                if w not in title and w not in stop_words:
                    title.append(w)

        title = " " . join(title)
        print("Selected Title:" , title , "\n")
        entities[cluster_id] = title

        ctr += 1
        #if ctr > 100:
        #    break

    df = pd.read_csv(
        dataset_path + dataset_name + '_nolabels.csv',
        names=['Product ID', 'Product Title', 'Vendor ID', 'Cluster ID', 'Cluster Label', 'Category ID',
               'Category Label'])

    out_lst : List[Any] = [None] * df.shape[0]
    x = 0
    for n_row, row in df.iterrows():
        out_lst[x] = [row['Product ID'], row['Product Title'], row['Vendor ID'], row['Cluster ID'],
                      entities[row['Cluster ID']], row['Category ID'], row['Category Label']]
        x+=1
    new_df = pd.DataFrame(out_lst)
    new_df.to_csv(dataset_path + dataset_name + ".csv", index=False)
        #print("Synthetic title:", title)
        #print()
        #ctr+= 1
        #if ctr > 100:
        #    break