import pandas as pd
from sentence_transformers import util


class Entity:
    def __init__(self, entity_id : str, fadrel_id : int):
        """
        Initialize a FaDReL Entity object.

        Args:
            entity_id (str): The entity ID (as it was provided by the training set).
            fadrel_id (int): The entity ID (internal FaDREL ID - zero-based integer).
        """
        self.id = entity_id
        self.fadrel_id = fadrel_id

        self.num_matching_records = 0
        self.matching_records = []

    def add_record(self, rec_id : int) -> None:
        """
        Add a record to the cluster of the Entity object.

        Args:
            rec_id (int): The ID of the record to add
        """
        self.matching_records.append(rec_id)
        self.num_matching_records += 1

    def compute_clustroids(self, records : list, method='max-length') -> None:
        """
        Compute the clustroid records of the Entity's cluster
        Identify the elements having the highest similarity with the other cluster elements.

        Args:
            records (list):
                The table of all records. It is used to retrieve additional information (titles, embeddings, etc.) about the matching records.
            method (str):
                The method to use to compute the clustroid records. Valid options are:
                    * 'max-length': The cluster is represented by the records with the longest titles
                    * 'min-length': The cluster is represented by the records with the shortest titles
                    * 'max-similarity': The cluster is represented by the records with the highest similarity
        """
        if method == 'max-similarity':
            similarities = [None] * self.num_matching_records
            i = 0
            for x1 in self.matching_records:
                sum_sim = sum([util.cos_sim(records[x1].embedding, records[x2].embedding).numpy()[0][0]
                               for x2 in self.matching_records])
                # similarities[i] = [x1, records[x1].id, records[x1].title, records[x1].entity_id, sum_sim]
                similarities[i] = [x1, sum_sim]
                i += 1
        elif method == 'max-length':
            similarities = [[x1, len(records[x1].title)] for x1 in self.matching_records]
        elif method == 'min-length':
            max_title_length = max([len(records[x1].title) for x1 in self.matching_records]) + 1
            similarities = [[x1, max_title_length - len(records[x1].title)] for x1 in self.matching_records]

        # clustroids = pd.DataFrame(similarities, columns=['recordID', 'recId', 'recordTitle', 'EntityID', 'sum_sim'])
        clustroids = pd.DataFrame(similarities, columns=['recordID', 'sum_sim'])
        clustroids = clustroids.sort_values(by=['sum_sim'], ascending=False)

        # In the end, we have the records of this entity sorted in decreasing similarity order. The element at
        # position 0 indicates the clustroid.
        self.matching_records = clustroids['recordID'].tolist()

    def display(self, show_embeddings : bool=False, show_contents : bool=True) -> None:
        """
        Display the Entity object properties

        Args:
            show_embeddings (bool): Whether to show the embeddings of the titles of the entities of the cluster.
            show_contents (bool): Whether to show the contents (entities) of the cluster.
        """

        print("================================================================")
        print("Entity ID: ", self.id, "(FaDReL ID:", self.fadrel_id,
              ") - Num Matching Records:", self.num_matching_records)

        if show_contents:
            self.display_matching_records(show_embeddings=show_embeddings)

        print()

    def display_matching_records(self, show_embeddings : bool=False) -> None:
        """
        Display the records in the Entity's cluster.

        """
        print("Num Matching Records:", self.num_matching_records)
        for e in self.matching_records:
            print(e)
