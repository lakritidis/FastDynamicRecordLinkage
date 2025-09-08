import numpy as np
import pandas as pd

import torch

from tqdm import tqdm

from sentence_transformers import util


class SentencePairBuilder:

    def __init__(self, search_min_threshold : int=0, search_max_threshold: int=1, pairs_path : str='',
                 num_neg_pairs_labels : int=3, num_pos_pairs_titles : int=3, num_neg_pairs_titles : int=0,
                 text_vectorizer =None, random_state=0) -> None:

        """
        Initialize the SentencePairBuilder of FaDReL

        Args:
            search_min_threshold (int): Minimum similarity threshold for a pair to be considered
            search_max_threshold (int): Maximum similarity threshold for a pair to be considered
            pairs_path (str): The path of the output files
            num_neg_pairs_labels (int): The number of negative pairs (with the labels) to train the seq. classifier
            num_pos_pairs_titles (int): The number of positive pairs (with other titles) to train the seq. classifier
            num_neg_pairs_titles (int): The number of negative pairs (with other titles) to train the seq. classifier
            random_state: The random state to use for training.
        """

        self.search_max_threshold = search_max_threshold,
        self.search_min_threshold = search_min_threshold
        self.random_state = random_state
        self.pairs_path = pairs_path
        self.num_neg_pairs_labels = num_neg_pairs_labels
        self.num_pos_pairs_titles = num_pos_pairs_titles
        self.num_neg_pairs_titles = num_neg_pairs_titles
        self.text_vectorizer = text_vectorizer

    def _find_similar_labels(self, record_title : str, record_title_embedding : torch.Tensor, entities : dict,
                             entity_label_inv_index : dict, clustroids : bool = False) -> np.array:
        """
        Given an input title ``entity_title``, return the ``self.num_neg_pairs_labels+1`` most similar entity labels.

        This is efficiently performed in two steps:
          1. Construct a set of candidate labels. A label is considered to be candidate, if it has at least one
             word in common with ``entity_title``. The inverted index significantly accelerates this process and
             returns a subset of candidate labels.
          2. Compute the similarities (cos sim between the sentence embeddings) between entity_title and every label
             in the subset of candidates.

        Args:
            record_title (str):
                The title for which we search the most similar entity labels.
            record_title_embedding (torch.Tensor):
                The vector representation (aka sentence embedding) of the record title.
            entities (dict(Entity)):
                A list of ``Entity`` objects - The pool from which the similar labels are retrieved
            entity_label_inv_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the labels of the input entities.
            clustroids (bool): Involve the entity cluster clustroids

        Returns:
            An np.array with the ``self.num_neg_pairs_labels+1`` most similar entity labels.
        """

        # step 1: Find the candidate labels
        candidate_entity_ids = []
        for w in record_title.split():
            if w in entity_label_inv_index:
                inverted_list = entity_label_inv_index[w]
                candidate_entity_ids.extend(inverted_list)

        candidate_entity_ids = np.unique(candidate_entity_ids).tolist()
        candidate_entity_emb = [entities[x].label_embedding for x in candidate_entity_ids]
        candidate_entity_lab = [entities[x].label for x in candidate_entity_ids]
        candidate_entity_clustroids = None
        if clustroids:
            candidate_entity_clustroids = [entities[x].clustroids.iloc[0, 0] for x in candidate_entity_ids]

        # Step 2: Find the most similar labels
        similarity = [util.cos_sim(record_title_embedding, embed).numpy()[0][0] * 100 for embed in candidate_entity_emb]
        similarity = np.array(similarity)

        # Search for entities with similarity in the range [min_t, max_t] - Similar entities are the candidates for
        # mining negative samples.
        similar_idx = np.where((similarity <= self.search_max_threshold) & (similarity >= self.search_min_threshold))
        if not similar_idx[0].shape[0]:
            similar_idx = np.where((similarity < self.search_min_threshold) & (similarity >= 0))

        similar_idx = similar_idx[0]

        similarity = np.take(similarity, similar_idx, axis=0).reshape(-1, 1)
        search_pool = np.take(candidate_entity_lab, similar_idx, axis=0).reshape(-1, 1)

        if clustroids:
            respective_clustroids = np.take(candidate_entity_clustroids, similar_idx, axis=0).reshape(-1, 1)
            search_pool = np.concatenate((search_pool, respective_clustroids, similarity), axis=1)
        else:
            search_pool = np.concatenate((search_pool, similarity), axis=1)

        # Sort the data in descending order
        search_pool = search_pool[search_pool[:, -1].argsort()][::-1]

        return search_pool[:self.num_neg_pairs_labels + 1]

    def _find_similar_titles(self, entity_embed : torch.Tensor, entity_matching_records : list) -> np.array:
        """
        Given the embedding of input title ``entity_embed``, return the ``self.num_pos_pairs_titles`` most similar
        entity titles.

        Args:
            entity_embed (torch.Tensor): The vector representation (embedding) of the entity title.
            entity_matching_records (list): A list with the record titles of the cluster where ``entity_embed`` belongs.

        Returns:
            An np.array with the ``self.num_pos_pairs_titles`` most similar entity titles.
        """

        titles_emb = [entity_matching_records[x] for x in entity_matching_records]
        titles = [x for x in entity_matching_records]

        similarity = [util.cos_sim(entity_embed, embed).numpy()[0][0] * 100 for embed in titles_emb]
        similarity = np.array(similarity)

        similar_idx = np.where((similarity < 100) & (similarity > 50))
        similarity = np.take(similarity, similar_idx, axis=0).reshape(-1, 1)

        search_pool = np.take(titles, similar_idx, axis=0).reshape(-1, 1)

        # Concat the ratio to the meta values of search pool
        search_pool = np.concatenate((search_pool, similarity), axis=1)

        # Sort the data in descending order
        search_pool = search_pool[search_pool[:, -1].argsort()][::-1]

        return search_pool[:self.num_pos_pairs_titles]

    def build_training_pairs(self, entities : dict, label_inverted_index : dict) -> pd.DataFrame:
        """
        Construct the training pairs for the training the Sequence Classifier.

        For each product in the training set:
          1. Construct one positive training pair with the label of the corresponding entity.
          2. Construct many negative training pairs from similar, but non-matching entity labels.
          3. Construct many positive training pairs with titles of other records in the same cluster/entity.

        Args:
            entities (dict):
                An list (in the form of a dict) that has been (pre)built on the clusters of the input entities.
            label_inverted_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the labels of the input clusters.

        Returns:
            A Pandas DataFrame containing the training pairs.
        """

        # Compute the cluster clustroids (will be used for positive/negative sampling with other record titles)
        for entity_id in entities:
            entity = entities[entity_id]
            entity.compute_clustroids()

        pairs = []
        for entity_id in tqdm(entities):
            # Create a positive (record title, entity label, 1) triple
            entity = entities[entity_id]
            entity_label = entity.label
            entity_matching_records = entity.matching_records
            num_matching_records = entity.num_matching_records

            for record_title in entity_matching_records.keys():
                record_title_embedding = entity_matching_records[record_title]

                # POSITIVE SAMPLES
                # One (strictly) positive tuple of the type (record_title, entity_label, 1)
                # print(entity_title, ': Positive Label:', cluster_label)
                pair = [record_title, entity_label, 1]
                pairs.append(pair)

                '''
                # Adaptive number of positive/negative pairs
                if num_matching_records > 10:
                    self.num_pos_pairs_titles = 1
                    self.num_neg_pairs_labels = self.num_pos_pairs_titles + 1
                elif num_matching_records > 5:
                    self.num_pos_pairs_titles = 2
                    self.num_neg_pairs_labels = self.num_pos_pairs_titles + 1
                elif num_matching_records > 1:
                    self.num_pos_pairs_titles = num_matching_records
                    self.num_neg_pairs_labels = self.num_pos_pairs_titles + 1
                else:
                    self.num_pos_pairs_titles = 0
                    self.num_neg_pairs_labels = 12
                '''

                # Additional positive tuples can be constructed from the similar records titles inside the same cluster
                if self.num_pos_pairs_titles > 0:
                    if num_matching_records > 1:
                        # positive_entities = self._find_similar_titles(record_title_embedding, entity_matching_records)

                        positive_entities = entity.clustroids.iloc[0:self.num_pos_pairs_titles, :].to_numpy()
                        num_positive_entities = positive_entities.shape[0]

                        for n in range(num_positive_entities):
                            positive_title = positive_entities[n, 0]
                            # print(entity_title, ": Positive Title: ", positive_title, '\n', positive_entities)
                            if record_title != positive_title:
                                pair = [record_title, positive_title, 1]
                                pairs.append(pair)

                # NEGATIVE SAMPLES
                # Many negative (record_title, entity_label, 0): We can have many. We pick the clusters with labels
                # which are the most similar to the entity.
                if self.num_neg_pairs_titles > 0 or self.num_neg_pairs_labels > 0:
                    negative_samples =self._find_similar_labels(
                        record_title, record_title_embedding, entities, label_inverted_index, True)

                    for n in range(negative_samples.shape[0]):
                        if negative_samples[n, 0] != entity_label:
                            negative_label = negative_samples[n, 0]
                            pair = [record_title, negative_label, 0]
                            pairs.append(pair)

                            negative_title = negative_samples[n, 1]
                            if negative_title != negative_label:
                                pair = [record_title, negative_title, 0]
                                pairs.append(pair)

                            # print(entity_title, ": Negative Label: ", negative_label)
                            # print(entity_title, ": Negative Title: ", negative_title)

        pairs_df = pd.DataFrame(pairs, columns=['t1', 't2', 'y'])
        pairs_df.to_csv(self.pairs_path, header=True, index=False)

        return pairs_df

    def find_candidate_clusters(self, test_df : pd.DataFrame, entities : dict, label_inverted_index : dict,
                                title_column : str, entity_label_column : str, entity_id_column : str) -> pd.DataFrame:
        """
        For each title in the test set, form a set of similar labels (from the existing ones).

        Args:
            test_df (pd.DataFrame): A Pandas DataFrame containing the test data (the entities to be matched).
            entities (dict):
                A list (in the form of a dict) that has been (pre)built on the input entities.
            label_inverted_index (dict):
                An inverted index (in the form of a dict) that has been (pre)built on the labels of the input clusters.
            title_column (str): column name of the record titles.
            entity_label_column (str): column name of the entity labels.
            entity_id_column (str): column name of the entity IDs.

        Returns:
            A Pandas DataFrame containing the testing pairs.
        """
        query_set = {}
        for index, row in tqdm(test_df.iterrows()):
            entity_title = row[title_column]
            cluster_label = row[entity_label_column]
            cluster_id = row[entity_id_column]

            entity_embed = self.text_vectorizer.encode([entity_title], convert_to_tensor=False)

            similar_candidate_labels = self._find_similar_labels(
                entity_title, entity_embed, entities, label_inverted_index, False)

            # print("Test Product title:", entity_title, " - Candidate Labels:\n\t", similar_candidate_labels)

            for label in similar_candidate_labels:
                y = 0
                if label[0] == cluster_label:
                    y = 1
                query_set[(entity_title, label[0])] = (label[1], y, cluster_label, cluster_id)

        query_list = [(key[0], key[1], val[0], val[1], val[2], val[3]) for key, val in query_set.items()]
        pairs_df = pd.DataFrame(query_list, columns=['t1', 't2', 'sim', 'y', 'real_label', 'real_entity_id'])

        pairs_df.to_csv(self.pairs_path, header=True, index=False)
        pairs_df = pd.read_csv(self.pairs_path, header=0)

        return pairs_df
