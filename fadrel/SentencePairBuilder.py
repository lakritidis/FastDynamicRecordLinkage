import numpy as np
import pandas as pd

from tqdm import tqdm

from sentence_transformers import util

from Record import Record

class SentencePairBuilder:

    def __init__(self, search_min_threshold : int=0, search_max_threshold: int=1, pairs_path : str='', triplets_path : str='',
                 num_neg_pairs : int=3, num_pos_pairs : int=3, text_vectorizer=None, random_state=0) -> None:

        """
        Initialize the SentencePairBuilder of FaDReL

        Args:
            search_min_threshold (int): Minimum similarity threshold for a pair to be considered
            search_max_threshold (int): Maximum similarity threshold for a pair to be considered
            pairs_path (str): The path of the output pairs
            triplets_path (str): The path of the output triplets
            num_neg_pairs (int): The number of negative pairs (with the labels) to train the seq. classifier
            num_pos_pairs (int): The number of positive pairs (with other titles) to train the seq. classifier
            random_state: The random state to use for training.
        """

        self.search_max_threshold = search_max_threshold,
        self.search_min_threshold = search_min_threshold
        self.random_state = random_state
        self.pairs_path = pairs_path
        self.triplets_path = triplets_path
        self.num_neg_pairs = num_neg_pairs
        self.num_pos_pairs = num_pos_pairs
        self.text_vectorizer = text_vectorizer

    # FUNCTIONS CALLED DURING PREPARATION/TRAINING PHASE
    def _find_negative_samples(self, record: Record, records: dict, inverted_index: dict, num_results: int) -> np.array:
        """
        Given an input title ``entity_title``, return the ``self.num_neg_pairs_labels+1`` most similar entity labels.

        This is efficiently performed in two steps:
          1. Construct a set of candidate labels. A label is considered to be candidate, if it has at least one
             word in common with ``entity_title``. The inverted index significantly accelerates this process and
             returns a subset of candidate labels.
          2. Compute the similarities (cos sim between the sentence embeddings) between entity_title and every label
             in the subset of candidates.

        Args:
            record (Record object):
                The Record object for which we search the most similar entity labels.
            records (dict(Record objects)):
                A dictionary of Record objects - The pool from which the negative samples are retrieved
            inverted_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the titles of the input records.
            num_results (int):
                The number of negative pairs to return.

        Returns:
            An np.array with the ``self.num_neg_pairs_labels+1`` most similar record titles.
        """

        # Step 1: Find the candidate records : non-matching records that have similar titles with the input record
        record_title = record.title
        record_entity_id = record.entity_id
        record_title_embedding = record.embedding

        candidate_record_ids = {}
        for w in record_title.split():
            if w in inverted_index:
                inverted_list = inverted_index[w]
                for candidate_record_id in inverted_list:

                    # Process the negative records only (not matching entities)
                    if records[candidate_record_id].entity_id != record_entity_id:
                        if candidate_record_id not in candidate_record_ids:
                            candidate_record_ids[candidate_record_id] = 1
                        else:
                            candidate_record_ids[candidate_record_id] += 1

        # Sort the dictionary of candidate IDs by the number of common words
        candidate_record_ids = dict(sorted(candidate_record_ids.items(), key=lambda item: item[1], reverse=True))
        candidate_record_ids = list(candidate_record_ids.keys())[0:200]
        # print("candidate_record_ids:", candidate_record_ids)

        # Step 2: Find the most similar labels
        candidate_embeddings = [records[r].embedding for r in candidate_record_ids]
        candidate_titles = [records[r].title for r in candidate_record_ids]
        similarities = [util.cos_sim(record_title_embedding, emb).numpy()[0][0] * 100 for emb in candidate_embeddings]
        similarities = np.array(similarities)

        # Search for records with similarity in the range [min_t, max_t] - Similar records are candidates
        # for mining hard negative samples.
        similar_idx = np.where((similarities <= self.search_max_threshold) & (similarities >= self.search_min_threshold))
        if not similar_idx[0].shape[0]:
            similar_idx = np.where((similarities < self.search_min_threshold) & (similarities >= 0))

        similar_idx = similar_idx[0]

        similarities = np.take(similarities, similar_idx, axis=0).reshape(-1, 1)
        search_pool = np.take(candidate_titles, similar_idx, axis=0).reshape(-1, 1)

        search_pool = np.concatenate((search_pool, similarities), axis=1)

        # Sort the data in descending order
        search_pool = search_pool[search_pool[:, -1].argsort()][::-1]

        return search_pool[:num_results]

    def build_training_pairs(self, entities : dict, records : dict, inverted_index : dict) -> (pd.DataFrame, pd.DataFrame):
        """
        Construct the training pairs for the training the Sequence Classifier.

        For each product in the training set:
          1. Construct one positive training pair with the label of the corresponding entity.
          2. Construct many negative training pairs from similar, but non-matching entity labels.
          3. Construct many positive training pairs with titles of other records in the same cluster/entity.

        Args:
            entities (dict):
                A dictionary that has been (pre)built on the clusters of the input entities.
            records (dict):
                A dictionary of Record objects that contains the input records.
            inverted_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the titles of the records.

        Returns:
            A tuple that contains two Pandas DataFrames with the training pairs and triplets.
        """
        pairs = []
        triplets = []
        for record in tqdm(records.values(), desc="\t\tBuilding training pairs & triplets..."):
            record_id = record.id
            record_title = record.title
            record_entity_id = record.entity_id  # The entity that this record matches with

            # POSITIVE PAIRS
            # Create positive pairs with the most representative records of the entity's cluster
            anchor = None
            if self.num_pos_pairs > 0:
                n_pos = 0

                # The other records that match this entity
                matching_records = entities[record_entity_id].matching_records
                # print(record_id, " - ", record_title, " - ", record_entity_id, " - ", matching_records)

                # The anchor of this entity is defined as the first element of its matching_records.
                # This first element could be the clustroid element, or the cluster's representative, leader, etc.
                anchor = records[matching_records[0]].title

                # Create positive pairs with the matching records.
                for matching_record_id in matching_records:
                    matching_record = records[matching_record_id]
                    if matching_record_id != record_id and matching_record.title != record_title:
                        pair = [record_title, matching_record.title, 1]
                        pairs.append(pair)
                        n_pos += 1
                        if n_pos >= self.num_pos_pairs:
                            break

            # NEGATIVE PAIRS
            if self.num_neg_pairs > 0:
                negative_samples = self._find_negative_samples(record, records, inverted_index, self.num_neg_pairs)

                for n in range(negative_samples.shape[0]):
                    if negative_samples[n, 0] != record_title:
                        negative_title = negative_samples[n, 0]
                        pair = [record_title, negative_title, 0]
                        pairs.append(pair)

                        # Create a training triplet: we do not have a negative label yet
                        if anchor is not None:
                            triplet = [anchor, record_title, negative_title]
                            # print("\tTriplet: ", triplet)
                            triplets.append(triplet)

        pairs_df = pd.DataFrame(pairs, columns=['t1', 't2', 'y'])
        pairs_df.to_csv(self.pairs_path, header=True, index=False)

        triplets_df = pd.DataFrame(triplets, columns=['anchor', 'positive', 'negative'])
        triplets_df.to_csv(self.triplets_path, header=True, index=False)

        return pairs_df, triplets_df

    # FUNCTIONS CALLED DURING QUERY PHASE
    def _find_similar_samples(self, record: Record, records: dict, inverted_index: dict, num_results: int) -> np.array:
        """
        Given an input title ``entity_title``, return the ``self.num_neg_pairs_labels+1`` most similar entity labels.

        This is efficiently performed in two steps:
          1. Construct a set of candidate labels. A label is considered to be candidate, if it has at least one
             word in common with ``entity_title``. The inverted index significantly accelerates this process and
             returns a subset of candidate labels.
          2. Compute the similarities (cos sim between the sentence embeddings) between entity_title and every label
             in the subset of candidates.

        Args:
            record (Record object):
                The Record object for which we search the most similar entity labels.
            records (dict(Record objects)):
                A dictionary of Record objects - The pool from which the negative samples are retrieved
            inverted_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the titles of the input records.
            num_results (int):
                The number of negative pairs to return.

        Returns:
            An np.array with the ``self.num_neg_pairs_labels+1`` most similar record titles.
        """

        # Step 1: Find the candidate records : non-matching records that have similar titles with the input record
        record_title = record.title
        record_title_embedding = record.embedding

        candidate_record_ids = {}
        for w in record_title.split():
            if w in inverted_index:
                inverted_list = inverted_index[w]
                for candidate_record_id in inverted_list:
                    if candidate_record_id not in candidate_record_ids:
                        candidate_record_ids[candidate_record_id] = 1
                    else:
                        candidate_record_ids[candidate_record_id] += 1

        # Sort the dictionary of candidate IDs by the number of common words
        candidate_record_ids = dict(sorted(candidate_record_ids.items(), key=lambda item: item[1], reverse=True))
        candidate_record_ids = list(candidate_record_ids.keys())[0:200]
        # print("candidate_record_ids:", candidate_record_ids)

        # Step 2: Find the records with most similar titles
        candidate_embeddings = [records[r].embedding for r in candidate_record_ids]
        similarities = [util.cos_sim(record_title_embedding, emb).numpy()[0][0] * 100 for emb in candidate_embeddings]
        similarities = np.array(similarities)

        # Search for records with similarity in the range [min_t, max_t] - Similar records are candidates
        # for mining hard negative samples.
        similar_idx = np.where((similarities <= self.search_max_threshold) & (similarities >= self.search_min_threshold))
        if not similar_idx[0].shape[0]:
            similar_idx = np.where((similarities < self.search_min_threshold) & (similarities >= 0))

        similar_idx = similar_idx[0]

        similarities = np.take(similarities, similar_idx, axis=0).reshape(-1, 1)
        search_pool = np.take(candidate_record_ids, similar_idx, axis=0).reshape(-1, 1)
        search_pool = np.concatenate((search_pool, similarities), axis=1)

        # Sort the data in descending order
        search_pool = search_pool[search_pool[:, -1].argsort()][::-1]

        return search_pool[:num_results]

    def find_candidate_records(self, test_df : pd.DataFrame, entities : dict, records : dict, inverted_index : dict,
                                title_column : str, entity_id_column : str) -> pd.DataFrame:
        """
        For each title in the test set, form a set of similar titles (from the existing ones).

        Args:
            test_df (pd.DataFrame):
                A Pandas DataFrame containing the test data (the entities to be matched).
            entities (dict):
                A dictionary that has been (pre)built on the clusters of the training entities.
            records (dict):
                A dictionary of Record objects that contains the training records.
            inverted_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the titles of the training records.
            title_column (str):
                The column name of the record titles.
            entity_id_column (str):
                The column name of the entity IDs.

        Returns:
            A Pandas DataFrame containing the testing pairs.
        """
        query_set = {}
        record_id = len(records)
        ncor = 0
        for index, row in tqdm(test_df.iterrows()):
            record_id += 1
            ground_truth_entity_id = row[entity_id_column]
            record_title = row[title_column]
            embedding = self.text_vectorizer.encode([record_title], convert_to_tensor=False)

            test_record = Record(rec_id=record_id, rec_title=record_title,
                                 entity_id=ground_truth_entity_id, embedding=embedding)

            similar_candidate_ids = self._find_similar_samples(record=test_record, records=records,
                                                               inverted_index=inverted_index,
                                                               num_results=self.num_pos_pairs)

            predicted_entity_id_0 = records[similar_candidate_ids[0][0]].entity_id

            # print("Record ID:", record_id, " - Title:", record_title)
            candidate_entities = {}
            for n in range(similar_candidate_ids.shape[0]):
                rid = similar_candidate_ids[n][0]
                sim = similar_candidate_ids[n][1]
                predicted_entity_id = records[rid].entity_id
                if predicted_entity_id not in candidate_entities:
                    candidate_entities[predicted_entity_id] = 1
                else:
                    candidate_entities[predicted_entity_id] += 1

                # print(rid, "-", records[rid].title, " == ", similar_candidate_ids[n][1])
                '''
                if predicted_entity_id == ground_truth_entity_id:
                    ncor += 1
                    if n != 0:
                        print("Correct found at position %d" % n)
                    break
                '''
            candidate_entities = dict(sorted(candidate_entities.items(), key=lambda item: item[1], reverse=True))
            predicted_entity_id = next(iter(candidate_entities))

            if predicted_entity_id_0 == ground_truth_entity_id:
                ncor += 1

            '''
            # print("Test Product title:", entity_title, " - Candidate Labels:\n\t", similar_candidate_labels)
            for label in similar_candidate_labels:
                y = 0
                if label[0] == cluster_label:
                    y = 1
                query_set[(entity_title, label[0])] = (label[1], y, cluster_label, cluster_id)
            '''
        print("Num correct:", ncor, " / ", test_df.shape[0])
        '''
        query_list = [(key[0], key[1], val[0], val[1], val[2], val[3]) for key, val in query_set.items()]
        pairs_df = pd.DataFrame(query_list, columns=['t1', 't2', 'sim', 'y', 'real_label', 'real_entity_id'])

        pairs_df.to_csv(self.pairs_path, header=True, index=False)
        pairs_df = pd.read_csv(self.pairs_path, header=0)

        return pairs_df
        '''

    def find_candidate_clustroids(self, test_df : pd.DataFrame, entities : dict, records : dict, inverted_index : dict,
                                  title_column : str, entity_id_column : str) -> pd.DataFrame:
        """
        For each title in the test set, form a set of similar labels (from the existing clustroids).

        Args:
            test_df (pd.DataFrame):
                A Pandas DataFrame containing the test data (the entities to be matched).
            entities (dict):
                A dictionary that has been (pre)built on the clusters of the training entities.
            records (dict):
                A dictionary of Record objects that contains the training records.
            inverted_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the titles of the training records.
            title_column (str):
                The column name of the record titles.
            entity_id_column (str):
                The column name of the entity IDs.

        Returns:
            A Pandas DataFrame containing the testing pairs.
        """
        query_set = {}
        record_id = len(records)
        ncor = 0
        for index, row in tqdm(test_df.iterrows()):
            record_id += 1
            ground_truth_entity_id = row[entity_id_column]
            record_title = row[title_column]
            embedding = self.text_vectorizer.encode([record_title], convert_to_tensor=False)

            test_record = Record(rec_id=record_id, rec_title=record_title,
                                 entity_id=ground_truth_entity_id, embedding=embedding)

            cand_ids = np.array([e for e in entities.keys()]).reshape(-1, 1)
            cand_titles = np.array([records[entities[e].matching_records[0]].title for e in entities.keys()]).reshape(-1, 1)
            cand_embeddings = [records[entities[e].matching_records[0]].embedding for e in entities.keys()]

            similarities = [util.cos_sim(test_record.embedding, emb).numpy()[0][0] * 100 for emb in cand_embeddings]
            similarities = np.array(similarities).reshape(-1, 1)

            search_pool = np.concatenate((cand_ids, cand_titles, similarities), axis=1)
            search_pool = search_pool[search_pool[:, -1].argsort()][::-1]

            print("Record title:", record_title, " == Ground Truth:", ground_truth_entity_id)
            print("Selected clustroid:", search_pool[0])
            if search_pool[0][0] == ground_truth_entity_id:
                print("CORRECT")
                ncor += 1
            else:
                print("INCORRECT == ", search_pool[0][0])
            print()
        print("Num correct:", ncor, " / ", test_df.shape[0])
