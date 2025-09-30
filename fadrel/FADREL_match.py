import os
import time
import pickle

import pandas as pd
import FADREL_prepare

from sentence_transformers import SentenceTransformer

from SentencePairClassifier import SentencePairClassifier
from SentencePairBuilder import SentencePairBuilder
from MatchMaker import MatchMaker


class FADRELMatchingPhase:
    def __init__(self, dataset_name : str, method_name : str, entity_id_col : str, record_title_col : str, paths : dict,
                 max_emb_len : int = 128, epochs : int = 5, batch_size : int = 16, evaluate : bool = True,
                 random_state=0) -> None:
        """
        Initialize the FaDReL matching phase pipeline

        Args:
            dataset_name (str): name of the dataset
            method_name (str) : name of the method (used to describe the results)
            entity_id_col (str): column name of the entity IDs
            record_title_col (str): column name of the record titles
            paths (dict): The paths of the output files
            max_emb_len (int): The maximum length of the title/label sentence embeddings.
            epochs (int): The number of epochs to train the classification model.
            batch_size (int): The batch size to train the classification model.
            evaluate (bool): Whether to evaluate the matching performance (requires ground truth entities).
            random_state: The random state to use for testing.
        """
        self.dataset_name = dataset_name
        self.method_name = method_name

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
        self.query_path = paths['query_path']

        self.max_emb_len = max_emb_len
        self.epochs = epochs
        self.batch_size = batch_size

        self.classifier = None
        self.entities = {}
        self.records = {}
        self.inv_index = None

        # If a Sentence Transformer has been previously fine-tuned, load it. Otherwise, use the pretrained MiniLM model.
        if os.path.isdir(self.vectorizer_path):
            print("\t\t\tA Pre-existing fine-tuned Sentence Transformer model was found and loaded", flush=True)
            self.text_vectorizer = SentenceTransformer(model_name_or_path=self.vectorizer_path)
        else:
            self.text_vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.eval = evaluate

    def initialize(self) -> None:
        """
        Initialization of the FaDReL matching phase.
        Load the sequence classification model, the cluster data and the inverted index.
        If any of these elements is missing, abort.
        """
        # Read a fine-tuned sequence classifier from a file
        if os.path.isdir(self.classifier_path):
            print("\t\tFound a pre-trained Sequence Classifier, loading...")
            self.classifier = SentencePairClassifier(
                model_name=self.classifier_path, model_path=self.classifier_path, max_emb_length=self.max_emb_len,
                epochs=self.epochs, batch_size=self.batch_size)
        else:
            print("\t\tNo pre-trained Classifier was found, aborting...")
            self.classifier = None
            #  exit()

        # Load the entities from the file that has been created during the preparation phase
        if os.path.isfile(self.entities_data_path):
            print("\t\tFound an entities' data file, loading...")
            data_file = open(self.entities_data_path, 'rb')
            self.entities = pickle.load(data_file)
            data_file.close()
        else:
            print("\t\tNo entities' data file was found, aborting...")
            exit()

        # Load the records from the file that has been created during the preparation phase
        if os.path.isfile(self.records_data_path):
            print("\t\tFound a records' data file, loading...")
            data_file = open(self.records_data_path, 'rb')
            self.records = pickle.load(data_file)
            data_file.close()
        else:
            print("\t\tNo records' data file was found, aborting...")
            exit()

        # Read the inverted index of the cluster labels from a file
        if os.path.isfile(self.inverted_index_path):
            print("\t\tFound an inverted file, loading...")
            data_file = open(self.inverted_index_path, 'rb')
            self.inv_index = pickle.load(data_file)
            data_file.close()
        else:
            print("\t\tNo inverted file was found, aborting...")
            exit()

    def build_query_set(self, test_df : pd.DataFrame) -> pd.DataFrame:
        """
        Build the query set.
        For each unknown record title, quickly form a set of candidate entities.
        """

        # Load/Create the query set for the Sequence Classification model. The Query Set is created as follows:
        # For each title t in the test set Te, find a set L of (similar) candidate cluster labels.
        # Then, create all possible (t in Te, l in L) pairs and store them in the query set.
        if os.path.isfile(self.query_path):
            print("\t\tPre-existing query pairs were found and loaded.")
            query_set = pd.read_csv(self.query_path, header=0)

        else:
            # Create the Query set:
            pair_builder = SentencePairBuilder(search_max_threshold=101, search_min_threshold=30,
                                               pairs_path=self.query_path, triplets_path=self.query_path,
                                               num_neg_pairs=0,
                                               num_pos_pairs=self.batch_size,
                                               text_vectorizer=self.text_vectorizer,
                                               random_state=self.random_state)
            query_set = pair_builder.find_candidate_records(test_df, entities=self.entities, records=self.records,
                                                            inverted_index=self.inv_index,
                                                            title_column=self.title_column,
                                                            entity_id_column=self.entity_id_column)
            #query_set = pair_builder.find_candidate_clustroids(test_df, entities=self.entities, records=self.records,
            #                                                   inverted_index=self.inv_index,
            #                                                   title_column=self.title_column,
            #                                                   entity_id_column=self.entity_id_column)

        return query_set

    def run(self, test_df : pd.DataFrame) -> None:
        """
        Runs the matching phase and performs record linkage

        Args:
            test_df (pd.DataFrame) : A DataFrame containing the records for which we search for a matching entity.
        """
        print("\n=== Starting FADREL matching phase...", flush=True)
        print("\tInitializing...", flush=True)
        start_time_init = time.time()

        self.initialize()
        # for c in self.entities:
        #    self.entities[c].display(show_embeddings=True, show_contents=True)
        test_df[self.title_column] = test_df[self.title_column].map(FADREL_prepare.preprocess_text)
        end_time_init = time.time()
        duration = end_time_init - start_time_init
        print("\tInitialization completed in %.3f sec" % duration, flush=True)

        print("\tBuilding query/test pairs...", flush=True)
        start_time_init = time.time()
        query_set = self.build_query_set(test_df)
        end_time_init = time.time()
        duration = end_time_init - start_time_init
        print("\tCompleted in %.3f sec" % duration, flush=True)

        #mm = MatchMaker(classifier=self.classifier, batch_size=self.batch_size, max_length=self.max_emb_len,
        #                text_vectorizer=self.text_vectorizer, method_name=self.dataset_name + "_" + self.method_name,
        #                evaluate=self.eval)

        # a = datetime.datetime.now().replace(microsecond=0)
        #mm.perform_matching(query_set, test_df, self.entities, self.title_column, self.label_column, self.label_id_column)
        # b = datetime.datetime.now().replace(microsecond=0)
        # print("Time:", b-a)
        # break
