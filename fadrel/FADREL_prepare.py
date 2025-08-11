import os
import time
import pickle

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from SentencePairClassifier import SentencePairClassifier
from SentencePairBuilder import SentencePairBuilder
from Entity import Entity


def preprocess_text(text):
    """
    Clean the white space and lowercase the text
    """
    return " ".join(text.split()).lower()


class FADRELPreparationPhase:
    def __init__(self, dataset_name : str, entity_id_col : str, entity_label_col : str, title_col : str, paths : dict,
                 max_emb_len : int = 128, epochs : int = 3, batch_size : int = 16, num_neg_pairs_labels : int = 1,
                 num_pos_pairs_titles: int = 1, num_neg_pairs_titles: int = 0, random_state = 0) -> None:
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
            random_state: The random state to use for training.
        """
        self.dataset_name = dataset_name

        self.title_column = title_col         # The column that stores the record title
        self.entity_id_column = entity_id_col   # The column that stores the entity ID
        self.entity_label_column = entity_label_col
        self.random_state = random_state

        self.classifier_path = paths['bert_path']
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
        self.text_vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def create_training_pairs(self) -> pd.DataFrame:
        """
        Create `(string 1, string 2, flag)` tuples for training the Sequence classifier:
           - `string 1`: title of a record
           - `string 2`: title of another record, or label of an entity
           - `flag`: 0 (the strings don't match - the pair is negative), or 1 (the strings match - the pair is positive)

        """
        # If the siamese pairs for training the model have been previously created for this fold, load them.
        if os.path.isfile(self.training_pairs_path):
            if os.path.isdir(self.classifier_path):
                training_pairs = None
                print("Pre-existing training pairs were found, but a classifier model already exists.", flush=True)
            else:
                training_pairs = pd.read_csv(self.training_pairs_path, header=0)
                print("Pre-existing training pairs were found and loaded.", flush=True)
        else:
            # Otherwise, create the siamese pairs to train the model
            print("Building training pairs...", flush=True)

            pair_builder = SentencePairBuilder(search_max_threshold=99, search_min_threshold=30,
                                               pairs_path=self.training_pairs_path,
                                               num_neg_pairs_labels=self.num_neg_pairs_labels,
                                               num_pos_pairs_titles=self.num_pos_pairs_titles,
                                               num_neg_pairs_titles=self.num_neg_pairs_titles,
                                               text_vectorizer=None,
                                               random_state=self.random_state)

            training_pairs = pair_builder.build_pairs(entities=self.entities, label_inverted_index=self.label_inv_index)

        return training_pairs

    def fine_tune_classifier(self) -> None:
        """
        Fine-tune the Sequence Classifier model
        """

        # If a BertForSequenceClassification model has been previously trained on the data of this fold, load it.
        if os.path.isdir(self.classifier_path):
            print("Found a pre-trained BERT Sequence Classifier.")
            self.classifier = None

        else:
            # Otherwise, train a new BertForSequenceClassification model on the data of this fold.
            self.classifier = SentencePairClassifier(
                model_name='bert-base-uncased', model_path=self.classifier_path,
                max_emb_length=self.max_emb_len, epochs=self.epochs, batch_size=self.batch_size)

            training_pairs = self.create_training_pairs()

            # Train the model with the siamese pairs
            print("Training BERT Sequence Classifier...")
            self.classifier.train(training_pairs)

    def initialize(self, train_df: pd.DataFrame) -> None:
        """
        Offline initialization phase that creates:
          1. The list of entities ``[(entity_id, entity_label, entity_label_embedding)]``
          2. The inverted index on the entity labels
          3. The list of records, grouped into their respective entities

        Args:
            train_df: Pandas DataFrame containing the training data
        """

        print("Initializing...")
        if os.path.isfile(self.label_index_path) and os.path.isfile(self.cluster_data_path):
            print("\tFound entity labels and their inverted index, aborting...")
            return None

        # Get the indexes of the data columns
        entity_id_column_idx = train_df.columns.get_loc(self.entity_id_column)
        entity_label_column_idx = train_df.columns.get_loc(self.entity_label_column)

        # Preprocess the text in the cluster labels and entity titles
        train_df[self.entity_label_column] = train_df[self.entity_label_column].apply(lambda x: preprocess_text(x))

        # Get the unique cluster IDs. Retrieve the corresponding labels and generate their sentence embeddings.
        entity_ids, indices = np.unique(train_df.loc[:, self.entity_id_column].to_numpy(), return_index=True)
        entity_labels = train_df.iloc[indices, entity_label_column_idx].to_numpy()

        print("\tComputing sentence (S-BERT) embeddings...", flush=True)
        entity_label_embeddings = self.text_vectorizer.encode(entity_labels, convert_to_tensor=False)

        # Create the list of entities: self.entities and their inverted index: self.label_inv_index. Both of them
        # are required to create the training positive and negative pairs. They are also required during the test
        # phase, so we construct them here (unless they exist already).
        print("\tIndexing...", flush=True)
        num_unique_clusters = indices.shape[0]
        for u in range(num_unique_clusters):
            entity_id = entity_ids[u]
            entity_label = entity_labels[u]
            entity_embedding = entity_label_embeddings[u]

            self.entities[entity_id] = Entity(
                entity_id=entity_id, entity_label=entity_label, label_embedding=entity_embedding)
            self.num_entities += 1
            # self.entities[cluster_id].display(True, True)

            # The inverted index on cluster labels
            label_words = entity_label.split()
            for word in label_words:
                if word not in self.label_inv_index:
                    self.label_inv_index[word] = [entity_id]
                else:
                    self.label_inv_index[word].append(entity_id)

        # Write the cluster labels (with their sentence embeddings) to a file
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
            train_df.self.title_column = train_df.self.title_column.apply(lambda x: preprocess_text(x))

            entity_titles = train_df.loc[:, self.title_column].to_numpy()
            title_embeddings = self.text_vectorizer.encode(entity_titles, convert_to_tensor=False)

            for n_row, row in enumerate(train_df.itertuples()):
                key = row[entity_id_column_idx + 1]
                entity_title = row[title_column_idx + 1]
                self.entities[key].add_record(title=entity_title, embedding=title_embeddings[n_row])

        return None

    def run(self, train_df : pd.DataFrame) -> None:
        """
        Run the preparation phase of FaDReL

        Args:
            train_df (pd.DataFrame): A Pandas DataFrame containing the training data

        """
        print("\n=== Starting Offline processing...", flush=True)

        start_time_init = time.time()
        self.initialize(train_df)
        end_time_init = time.time()
        duration = end_time_init - start_time_init
        print("Initialization completed in %.3f sec" % duration, flush=True)

        # for c in self.clusters:
        #    self.clusters[c].display(show_embeddings=False, show_contents=True)

        start_time_trn = time.time()
        self.fine_tune_classifier()
        end_time_trn = time.time()
        duration = end_time_trn - start_time_trn
        print("Sequence classification model was loaded/trained in %.3f sec" % duration, flush=True)
