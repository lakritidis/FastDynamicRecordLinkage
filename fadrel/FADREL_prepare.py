import os
import time
import pickle
from typing import Any, List

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import ContrastiveLoss, SiameseDistanceMetric

from SentencePairClassifier import SentencePairClassifier
from SentencePairBuilder import SentencePairBuilder
from Entity import Entity


def preprocess_text(s):
    """
    Clean the white space and lowercase the text
    """
    return ' '.join(s.split()).lower()


class FADRELPreparationPhase:
    def __init__(self, dataset_name : str, entity_id_col : str, entity_label_col : str, title_col : str, paths : dict,
                 max_emb_len : int = 128, epochs : int = 3, batch_size : int = 16, num_neg_pairs_labels : int = 1,
                 num_pos_pairs_titles: int = 1, num_neg_pairs_titles: int = 0, finetune_sbert: bool = False,
                 random_state = 0) -> None:
        """
        Initialize the FaDReL preparation pipeline

        Args:
            dataset_name (str): name of the dataset
            entity_id_col (str): column name of the label ids
            entity_label_col (str): column name of the label strings
            title_col (str): column name of the entity titles
            paths (dict): The paths of the output files
            max_emb_len (int): The maximum length of the title/label sentence embeddings.
            epochs (int): The number of epochs to train the classification model.
            batch_size (int): The batch size to train the classification model.
            num_neg_pairs_labels (int): The number of negative pairs (with the labels) to train the seq. classifier
            num_pos_pairs_titles (int): The number of positive pairs (with other titles) to train the seq. classifier
            num_neg_pairs_titles (int): The number of negative pairs (with other titles) to train the seq. classifier
            finetune_sbert (bool): Whether to finetune the SBERT model.
            random_state: The random state to use for training.
        """
        self.dataset_name = dataset_name

        self.title_column = title_col         # The column that stores the record title
        self.entity_id_column = entity_id_col   # The column that stores the entity ID
        self.entity_label_column = entity_label_col
        self.random_state = random_state

        self.classifier_path = paths['bert_path']
        self.vectorizer_path = paths['vectorizer_path']
        self.training_pairs_path = paths['train_path']
        self.label_index_path = paths['lab_index_path']
        self.cluster_data_path = paths['cl_data_path']

        self.max_emb_len = max_emb_len
        self.epochs = epochs
        self.batch_size = batch_size

        self.num_neg_pairs_labels = num_neg_pairs_labels
        self.num_pos_pairs_titles = num_pos_pairs_titles
        self.num_neg_pairs_titles = num_neg_pairs_titles

        self.num_entities = 0
        self.entities = {} # A dictionary of Entity objects

        self.label_inv_index = {} # The inverted index constructed with the entity labels

        self.classifier = None
        self.fine_tune_sbert = finetune_sbert
        self.text_vectorizer = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2')

    def create_training_pairs(self) -> pd.DataFrame:
        """
        Create `(string 1, string 2, flag)` tuples for training the Sequence classifier:
           - `string 1`: title of a record
           - `string 2`: title of another record, or label of an entity
           - `flag`: 0 (the strings don't match - the pair is negative), or 1 (the strings match - the pair is positive)

        """
        # If the siamese pairs for training the model have been previously created for this fold, load them.
        if os.path.isfile(self.training_pairs_path):
            training_pairs = pd.read_csv(self.training_pairs_path, header=0)
            print("\t\t\tPre-existing training pairs were found and loaded", flush=True)
        else:
            # Otherwise, create the siamese pairs to train the model
            pair_builder = SentencePairBuilder(search_max_threshold=70, search_min_threshold=30,
                                               pairs_path=self.training_pairs_path,
                                               num_neg_pairs_labels=self.num_neg_pairs_labels,
                                               num_pos_pairs_titles=self.num_pos_pairs_titles,
                                               num_neg_pairs_titles=self.num_neg_pairs_titles,
                                               text_vectorizer=None,
                                               random_state=self.random_state)

            training_pairs = pair_builder.build_training_pairs(entities=self.entities,
                                                               label_inverted_index=self.label_inv_index)

        # Fine-tune Sentence Transformer and create new training pairs
        if self.fine_tune_sbert:
            if os.path.isdir(self.vectorizer_path):
                print("\t\t\tA Pre-existing fine-tuned Sentence Transformer model was found and loaded", flush=True)
                self.text_vectorizer = SentenceTransformer(model_name_or_path=self.vectorizer_path)
            else:
                print("\t\t\tFine-tuning the Sentence Transformer model...", flush=True)
                start_time = time.time()
                train_examples: List[Any] = [None] * training_pairs.shape[0]
                for n_row, row in enumerate(training_pairs.itertuples()):
                    train_examples[n_row] = InputExample(texts=[row[1], row[2]], label=row[3])

                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
                model = SentenceTransformer('all-MiniLM-L6-v2')
                train_loss = ContrastiveLoss(model=model, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
                                             margin=0.5)

                self.text_vectorizer.fit(
                    train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100,
                    show_progress_bar=False, save_best_model=True, output_path=self.vectorizer_path
                )
                end_time = time.time()
                duration = end_time - start_time
                print("\t\t\tTraining time:", duration, flush=True)

            # Recompute the embeddings with the fine-tuned Sentence Transformer
            print("\t\t\tRecomputing the embeddings...", flush=True)
            for e in self.entities.keys():
                # Update the embeddings of the entity labels
                entity = self.entities[e]
                entity.label_embedding = self.text_vectorizer.encode([entity.label], convert_to_tensor=False)

                matching_records = entity.matching_records
                # Update the embeddings of the titles of the matching records
                for record_title in matching_records.keys():
                    matching_records[record_title] = self.text_vectorizer.encode([record_title], convert_to_tensor=False)


            # Create new training pairs
            pair_builder = SentencePairBuilder(search_max_threshold=70, search_min_threshold=30,
                                               pairs_path=self.training_pairs_path,
                                               num_neg_pairs_labels=self.num_neg_pairs_labels,
                                               num_pos_pairs_titles=self.num_pos_pairs_titles,
                                               num_neg_pairs_titles=self.num_neg_pairs_titles,
                                               text_vectorizer=None,
                                               random_state=self.random_state)

            training_pairs = pair_builder.build_training_pairs(entities=self.entities,
                                                               label_inverted_index=self.label_inv_index)
        return training_pairs

    def fine_tune_classifier(self) -> None:
        """
        Fine-tune the Sequence Classifier model
        """

        # If a BertForSequenceClassification model has been previously trained on the data of this fold, load it.
        if os.path.isdir(self.classifier_path):
            print("\t\tFound a pre-trained BERT Sequence Classifier, skipping fine-tuning", flush=True)
            self.classifier = None

        else:
            # Otherwise, train a new BertForSequenceClassification model on the data of this fold.
            self.classifier = SentencePairClassifier(
                model_name='bert-base-uncased', model_path=self.classifier_path,
                max_emb_length=self.max_emb_len, epochs=self.epochs, batch_size=self.batch_size)

            print("\t\tBuilding training pairs...", flush=True)
            start_time = time.time()
            training_pairs = self.create_training_pairs()
            end_time = time.time()
            duration = end_time - start_time
            print("\t\tCompleted in %.3f sec" % duration, flush=True)

            # Train the model with the siamese pairs
            print("\t\tTraining...", flush=True)
            start_time = time.time()
            self.classifier.train(training_pairs)
            end_time = time.time()
            duration = end_time - start_time
            print("\t\tCompleted in %.3f sec" % duration, flush=True)

    def initialize(self, train_df : pd.DataFrame) -> None:
        """
        Offline initialization phase that creates:
          1. The list of entities ``[(entity_id, entity_label, entity_label_embedding)]``
          2. The inverted index on the entity labels
          3. The list of records, grouped into their respective entities

        Args:
            train_df: Pandas DataFrame containing the training data
        """

        if os.path.isfile(self.label_index_path) and os.path.isfile(self.cluster_data_path):
            print("\t\tFound entity labels and their inverted index, aborting...")
            return None

        # Get the indexes of the data columns
        entity_id_column_idx = train_df.columns.get_loc(self.entity_id_column)
        entity_label_column_idx = train_df.columns.get_loc(self.entity_label_column)

        # Preprocess the text in the cluster labels and entity titles
        train_df[self.entity_label_column] = train_df[self.entity_label_column].map(preprocess_text)

        # Get the unique cluster IDs. Retrieve the corresponding labels and generate their sentence embeddings.
        entity_ids, indices = np.unique(train_df.loc[:, self.entity_id_column].to_numpy(), return_index=True)
        entity_labels = train_df.iloc[indices, entity_label_column_idx].to_numpy()

        print("\t\tComputing Sentence embeddings...", flush=True)
        entity_label_embeddings = self.text_vectorizer.encode(entity_labels, convert_to_tensor=False)

        # Create the dictionary of entities (self.entities) and their inverted index (self.label_inv_index).
        # Both structures are required to create the training positive and negative pairs. They are also
        # required during the test phase, so we construct them here (unless they already exist).
        print("\t\tIndexing...", flush=True)
        num_unique_clusters = indices.shape[0]

        for u in range(num_unique_clusters):
            entity_id = entity_ids[u]
            entity_label = entity_labels[u]
            entity_embedding = entity_label_embeddings[u]

            self.entities[entity_id] = Entity(entity_id=entity_id, fadrel_id=self.num_entities,
                                              entity_label=entity_label, label_embedding=entity_embedding)
            self.num_entities += 1
            # self.entities[cluster_id].display(True, True)

            # The inverted index on cluster labels
            label_words = entity_label.split()
            for word in label_words:
                if word not in self.label_inv_index:
                    self.label_inv_index[word] = [entity_id]
                else:
                    self.label_inv_index[word].append(entity_id)

        # Write the entity labels (with their sentence embeddings) to a file
        data_file = open(self.cluster_data_path, 'wb')
        pickle.dump(self.entities, data_file)
        data_file.close()

        # Write the labels inverted index to a file
        data_file = open(self.label_index_path, 'wb')
        pickle.dump(self.label_inv_index, data_file)
        data_file.close()

        # The following block adds the records to their corresponding entities. It also generates the sentence
        # embeddings of the record titles. The block is executed conditionally, if no BERTClassifier exists: we
        # only need it to create positive/negative pairs with the titles.
        if not os.path.isfile(self.training_pairs_path) and not os.path.isfile(self.classifier_path):
            title_column_idx = train_df.columns.get_loc(self.title_column)
            train_df[self.title_column] = train_df[self.title_column].map(preprocess_text)

            # Get the record titles
            record_titles = train_df.loc[:, self.title_column].to_numpy()
            title_embeddings = self.text_vectorizer.encode(record_titles, convert_to_tensor=False)

            for n_row, row in enumerate(train_df.itertuples()):
                key = row[entity_id_column_idx + 1]
                entity_title = row[title_column_idx + 1]

                # self.entities is a dictionary, but it acts as a hash table here. It quickly locates the entity
                # to match this record.
                self.entities[key].add_record(title=entity_title, embedding= title_embeddings[n_row])

        return None

    def run(self, train_df : pd.DataFrame) -> None:
        """
        Run the preparation phase of FaDReL

        Args:
            train_df (pd.DataFrame): A Pandas DataFrame containing the training data

        """
        print("\n=== Starting FADREL Preparation phase...", flush=True)

        print("\tInitializing...", flush=True)
        start_time_init = time.time()
        self.initialize(train_df)
        end_time_init = time.time()
        duration = end_time_init - start_time_init
        print("\tInitialization completed in %.3f sec" % duration, flush=True)

        # for c in self.clusters:
        #    self.clusters[c].display(show_embeddings=False, show_contents=True)

        print("\tFine-Tuning the Sequence Classifier...")
        start_time_trn = time.time()
        self.fine_tune_classifier()
        end_time_trn = time.time()
        duration = end_time_trn - start_time_trn
        print("\tThe Sequence Classifier was loaded/trained in %.3f sec" % duration, flush=True)
