import pandas as pd
import torch
from sentence_transformers import util


class Entity:
    def __init__(self, entity_id : str, entity_label : str, label_embedding : torch.Tensor):
        """
        Initialize a FaDReL Entity object.

        Args:
            entity_id (str): The entity ID.
            entity_label (str): The entity label.
            label_embedding (Torch.tensor): The embedding of the entity label.
        """
        self.id = entity_id
        self.label = entity_label
        self.label_embedding = label_embedding


        self.num_matching_records = 0
        self.matching_records = {}  # A dictionary of (Record Title -> Record Embedding) entries
        self.clustroids = None

    def add_record(self, title : str, embedding : torch.Tensor) -> None:
        """
        Add a record to the cluster of the Entity object.

        Args:
            title (str): The entity title
            embedding  (Torch.tensor): The embedding of the entity title
        """

        if title not in self.matching_records:
            self.matching_records[title] = embedding
            self.num_matching_records += 1

    def compute_clustroids(self) -> None:
        """
        Compute the clustroid records of the Entity's cluster
        Identify the elements having the highest similarity with the other cluster elements.
        """
        similarities = [None] * self.num_matching_records
        i = 0
        for e1 in self.matching_records:
            sum_sim = sum([util.cos_sim(self.matching_records[e1], self.matching_records[e]).numpy()[0][0] for e in self.matching_records])
            similarities[i] = [e1, sum_sim]
            i += 1

        self.clustroids = pd.DataFrame(similarities, columns=['entity', 'sum_sim'])
        self.clustroids = self.clustroids.sort_values(by=['sum_sim'], ascending=False)

    def display(self, show_embeddings : bool=False, show_contents : bool=True) -> None:
        """
        Display the Entity object properties

        Args:
            show_embeddings (bool): Whether to show the embeddings of the titles of the entities of the cluster.
            show_contents (bool): Whether to show the contents (entities) of the cluster.
        """

        print("================================================================")
        if show_embeddings:
            print("Entity ID: ", self.id, "- Label:", self.label, "- Num Matching Records:", self.num_matching_records,
                  "- Embedding:", self.label_embedding)
        else:
            print("Entity ID: ", self.id, "- Label:", self.label, "- Num Matching Records:", self.num_matching_records)

        if show_contents:
            self.display_matching_records(show_embeddings=show_embeddings)
        print()

    def display_clustroids(self):
        """
        Display the clustroid records of the Entity's cluster
        """
        print(self.clustroids)

    def display_matching_records(self, show_embeddings : bool=False) -> None:
        """
        Display the records in the Entity's cluster.

        Args:
            show_embeddings (bool): Whether to show the embeddings of the record titles.

        """
        print("Num Matching Records:", self.num_matching_records)
        for e in self.matching_records:
            if show_embeddings:
                print("\t", (e, self.matching_records[e]))
            else:
                print("\t", e)