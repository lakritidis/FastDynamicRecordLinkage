import os
import time
import pickle
import html
from typing import Any, List

import pandas as pd

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import ContrastiveLoss, SiameseDistanceMetric, TripletLoss

from SentencePairClassifier import SentencePairClassifier
from SentencePairBuilder import SentencePairBuilder

from Entity import Entity
from Record import Record


def preprocess_text(s):
    """
    Clean the white space and lowercase the text
    """
    s = html.unescape(s)
    #translator = str.maketrans('&', ' ')
    #s = s.translate(translator)
    ret =  ' '.join(s.split()).lower()
    return ret


class FADRELPreparationPhase:
    def __init__(self, dataset_name : str, entity_id_col : str, record_title_col : str, paths : dict,
                 max_emb_len : int = 128, epochs : int = 3, batch_size : int = 16, num_neg_pairs : int = 1,
                 num_pos_pairs : int = 1, finetune_sbert: bool = False, random_state = 0) -> None:
        """
        Initialize the FaDReL preparation pipeline

        Args:
            dataset_name (str): name of the dataset
            entity_id_col (str): column name of the entity IDs
            record_title_col (str): column name of the record titles
            paths (dict): The paths of the output files
            max_emb_len (int): The maximum length of the title/label sentence embeddings.
            epochs (int): The number of epochs to train the classification model.
            batch_size (int): The batch size to train the classification model.
            num_neg_pairs (int): The number of negative pairs to train the Sequence classifier
            num_pos_pairs (int): The number of positive pairs to train the Sequence classifier
            finetune_sbert (bool): Whether to finetune the SBERT model.
            random_state: The random state to use for training.
        """
        self.dataset_name = dataset_name

        self.title_column = record_title_col    # The column that stores the record title
        self.entity_id_column = entity_id_col   # The column that stores the entity ID
        self.random_state = random_state

        self.classifier_path = paths['bert_path']
        self.vectorizer_path = paths['vectorizer_path']
        self.training_pairs_file = paths['pairs_file']
        self.training_triplets_file = paths['triplets_file']
        self.inverted_index_path = paths['inverted_index_datafile']
        self.records_data_path = paths['records_datafile']
        self.entities_data_path = paths['entities_datafile']

        self.max_emb_len = max_emb_len
        self.epochs = epochs
        self.batch_size = batch_size

        self.num_neg_pairs = num_neg_pairs
        self.num_pos_pairs = num_pos_pairs

        self.num_entities = 0
        self.entities = {}  # A dictionary of Entity objects
        self.num_records = 0
        self.records = {}

        self.inv_index = {} # The inverted index constructed with the record titles

        self.classifier = None
        self.fine_tune_sbert = finetune_sbert
        self.text_vectorizer = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2')

    def create_training_pairs(self) -> pd.DataFrame:
        """
        Create `(string 1, string 2, flag)` tuples for training the Sequence classifier:
           - `string 1`: title of a record
           - `string 2`: title of another record, or label of an entity
           - `flag`: 0 (the strings don't match - the pair is negative), or 1 (the strings match - the pair is positive)

        Pipeline:
            1. Create positive and negative pairs with the pre-trained Sentence Transformer.
            2. Fine-tune the Sentence Transformer model and create new (better) training pairs.
            3. Recompute the embeddings of the record titles with the fine-tuned Sentence Transformer.
            4. Recompute the clustroids of the entity clusters.
            5. Create new positive and negative pairs with the fine-tuned Sentence Transformer.
        """
        # If the siamese pairs for training the model have been previously created for this fold, load them.
        if os.path.isfile(self.training_pairs_file):
            training_pairs = pd.read_csv(self.training_pairs_file, header=0)
            print("\t\t\tPre-existing training pairs were found and loaded", flush=True)
        else:
            # Step 1: Otherwise, create the siamese pairs to train the classification model. Initially, we use the
            # pretrained Sentence Transformer model.
            pair_builder = SentencePairBuilder(search_max_threshold=70, search_min_threshold=30,
                                               pairs_path=self.training_pairs_file, triplets_path=self.training_triplets_file,
                                               num_neg_pairs=self.num_neg_pairs, num_pos_pairs=self.num_pos_pairs,
                                               text_vectorizer=None,
                                               random_state=self.random_state)

            training_pairs, training_triplets = pair_builder.build_training_pairs(entities=self.entities,
                                                                                  records=self.records,
                                                                                  inverted_index=self.inv_index)

            # Step 2: Fine-tune the Sentence Transformer model and create new (better) training pairs.
            if self.fine_tune_sbert:
                if os.path.isdir(self.vectorizer_path):
                    print("\t\tA Pre-existing fine-tuned Sentence Transformer model was found and loaded", flush=True)
                    self.text_vectorizer = SentenceTransformer(model_name_or_path=self.vectorizer_path)
                else:
                    print("\t\tFine-tuning the Sentence Transformer model...", flush=True)
                    start_time = time.time()

                    # Fine-tuning the Sentence Transformer model with pairs and Contrastive Loss
                    train_pairs: List[Any] = [None] * training_pairs.shape[0]
                    for n_row, row in enumerate(training_pairs.itertuples()):
                        train_pairs[n_row] = InputExample(texts=[row[2], row[1]], label=row[3])
                    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=16)
                    train_loss = ContrastiveLoss(model=SentenceTransformer('all-MiniLM-L6-v2'),
                                                 distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin=0.5)

                    # Fine-tuning the Sentence Transformer model with triplets and Triplet Loss
                    #train_triplets: List[Any] = [None] * training_triplets.shape[0]
                    #for n_row, row in enumerate(training_triplets.itertuples()):
                    #    train_triplets[n_row] = InputExample(texts=[row[1], row[2], row[3]])
                    #train_dataloader = DataLoader(train_triplets, shuffle=True, batch_size=16)
                    #train_loss = TripletLoss(model=SentenceTransformer('all-MiniLM-L6-v2'),)

                    self.text_vectorizer.fit(
                        train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100,
                        show_progress_bar=False, save_best_model=True, output_path=self.vectorizer_path,
                        optimizer_params={'lr': 2e-5}, weight_decay=0.01
                    )

                    end_time = time.time()
                    duration = end_time - start_time
                    print("\t\tTraining time:", duration, flush=True)

                # Step 3: Re-encode the record titles with the fine-tuned Sentence Transformer
                print("\t\tRecomputing the embeddings...", flush=True)
                for r in self.records.keys():
                    # Update the embeddings of the entity labels
                    record = self.records[r]
                    record.embedding = self.text_vectorizer.encode([record.title], convert_to_tensor=False)

                # Write the records and their updated title embeddings to a file
                with open(self.records_data_path, 'wb') as data_file:
                    # noinspection PyTypeChecker
                    pickle.dump(self.records, data_file)
                    data_file.close()

                # Step 4: Recompute the clustroids of the entity clusters
                for e in self.entities:
                    self.entities[e].compute_clustroids(records=self.records, method='max-length')

                # Write the entities and their updated clustroids to a file
                with open(self.entities_data_path, 'wb') as data_file:
                    # noinspection PyTypeChecker
                    pickle.dump(self.entities, data_file)
                    data_file.close()

                # Step 5: Create new training pairs with the fine-tuned Sentence Transformer
                pair_builder = SentencePairBuilder(search_max_threshold=65, search_min_threshold=55,
                                                   pairs_path=self.training_pairs_file, triplets_path=self.training_triplets_file,
                                                   num_neg_pairs=self.num_neg_pairs, num_pos_pairs=self.num_pos_pairs,
                                                   text_vectorizer=self.text_vectorizer,
                                                   random_state=self.random_state)

                training_pairs, training_triplets = pair_builder.build_training_pairs(entities=self.entities,
                                                                                      records=self.records,
                                                                                      inverted_index=self.inv_index)
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

            training_pairs = self.create_training_pairs()

            # Train the model with the siamese pairs
            print("\t\tTraining...", flush=True)
            start_time = time.time()
            # self.classifier.train(training_pairs)
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

        if os.path.isfile(self.inverted_index_path) and os.path.isfile(self.records_data_path) and os.path.isfile(self.entities_data_path):
            print("\t\tFound record/entity data files and the inverted index, loading...")
            if not os.path.isfile(self.classifier_path) and not os.path.isfile(self.training_pairs_file):
                with (open(self.records_data_path, "rb")) as data_file:
                    while True:
                        try:
                            self.records = pickle.load(data_file)
                        except EOFError:
                            break

                with (open(self.entities_data_path, "rb")) as data_file:
                    while True:
                        try:
                            self.entities = pickle.load(data_file)
                        except EOFError:
                            break

                with (open(self.inverted_index_path, "rb")) as data_file:
                    while True:
                        try:
                            self.inv_index = pickle.load(data_file)
                        except EOFError:
                            break

            return None

        # Create the list of records (self.records) and the inverted index of the record titles (self.index).
        # All three structures are required to create the training positive and negative pairs.

        # The following block adds the records to their corresponding entities. It also generates the sentence
        # embeddings of the record titles. The block is executed conditionally, if no BERTClassifier exists: we
        # only need it to create positive/negative pairs with the titles.
        if not os.path.isfile(self.classifier_path) and not os.path.isfile(self.training_pairs_file):
            print("\t\tIndexing...", flush=True, end='')

            title_column_idx = train_df.columns.get_loc(self.title_column)
            entity_id_column_idx = train_df.columns.get_loc(self.entity_id_column)

            train_df[self.title_column] = train_df[self.title_column].map(preprocess_text)

            # Get the record titles and compute their embeddings with the Sentence Transformer
            record_titles = train_df.loc[:, self.title_column].to_numpy()
            title_embeddings = self.text_vectorizer.encode(record_titles, convert_to_tensor=False)

            for n_row, row in enumerate(train_df.itertuples()):
                self.num_records += 1

                entity_id = row[entity_id_column_idx + 1]
                record_title = row[title_column_idx + 1]

                # self.entities is a dictionary, but it acts as a hash table here. It quickly finds the entity
                # to accommodate this record.
                self.records[self.num_records] = Record(rec_id=self.num_records, rec_title=record_title,
                                                        entity_id=entity_id, embedding=title_embeddings[n_row])

                if entity_id not in self.entities.keys():
                    self.entities[entity_id] = Entity(entity_id=entity_id, fadrel_id=self.num_entities)
                    self.entities[entity_id].add_record(self.num_records)
                    self.num_entities += 1
                else:
                    self.entities[entity_id].add_record(self.num_records)

                # The inverted index is built with the record titles
                title_words = set(record_title.split())
                for word in title_words:
                    if word not in self.inv_index:
                        self.inv_index[word] = [self.num_records]
                    else:
                        self.inv_index[word].append(self.num_records)

            # Compute the cluster clustroids (will be used for positive/negative sampling with other record titles)
            for e in self.entities:
                self.entities[e].compute_clustroids(records=self.records, method='max-length')

            # Write the records to a file
            with open(self.records_data_path, 'wb') as data_file:
                # noinspection PyTypeChecker
                pickle.dump(self.records, data_file)
                data_file.close()

            # Write the entities to a file
            with open(self.entities_data_path, 'wb') as data_file:
                # noinspection PyTypeChecker
                pickle.dump(self.entities, data_file)
                data_file.close()

            # Write the labels inverted index to a file
            with open(self.inverted_index_path, 'wb') as data_file:
                # noinspection PyTypeChecker
                pickle.dump(self.inv_index, data_file)
                data_file.close()

            print(" completed", flush=True)
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
