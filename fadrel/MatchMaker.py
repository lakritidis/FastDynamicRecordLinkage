import pandas as pd
from tqdm import tqdm

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from SentencePairDataset import SentencePairDataset
from SentencePairClassifier import SentencePairClassifier
from FADREL_results import FADRELResult


class MatchMaker:
    def __init__(self, classifier : SentencePairClassifier, batch_size : int, max_length : int,
                 text_vectorizer : SentenceTransformer, method_name : str) -> None:
        """
        Initialize a MatchMaker object.

        Args:
            classifier (SentencePairClassifier): The BERT Classification model to be used for matching.
            batch_size (int): The batch size to be used during testing.
            max_length (int): The maximum embedding length of the BERT Classification model.
            method_name (str): The name of the entity matching method to be evaluated.
        """
        self._model = classifier.model
        self._tokenizer = classifier.tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._method_name = method_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_new_clusters = 0
        self.new_clusters = {}
        self.text_vectorizer = text_vectorizer

    def perform_matching(self, query_df:pd.DataFrame, test_df : pd.DataFrame, title_col : str, label_col : str) -> None:
        """
        Find the matching entities for the unknown input record titles. An input title may match one, or none of
        the existing entities. In the latter case, the algorithm maintains self.new_clusters to accommodate these
        titles.

        Args:
            query_df (pd.DataFrame) : A DataFrame that contains the candidate entities for each record title.
            test_df (pd.DataFrame) : A DataFrame that contains the unknown (unmatched) records
            title_col (str) : The column name of test_df that contains the titles of the unmatched records.
            label_col (str) : The column name of test_df that contains the ground truth entity labels of the records.
        """
        print("\nMatching the unlabeled entities...", flush=True)
        eval_dataset = SentencePairDataset(self._tokenizer, self._batch_size, self._max_length)
        unmatched_entities = []

        self._model.eval()

        # For each title in the query set, retrieve the candidate labels and create a dataloader.
        # We will compare each title with all candidate (i.e. most similar) cluster labels.
        test_records_df = test_df.loc[:, [title_col, label_col]]

        num_correct_labeled = 0
        num_correct_seen, num_incorrect_seen, num_correct_unseen, num_incorrect_unseen, num_actual_seen = 0, 0, 0, 0, 0

        # print(test_df.shape)
        for n_row, row in tqdm(enumerate(test_records_df.itertuples())):
            title = row[1]
            ground_truth_label = row[2]

            # Fetch the candidate cluster labels from the query set. The query set has been built in FADREL_match.py
            # and contains the labels that have at least one word in common with the entity title.
            df = query_df.loc[query_df['t1'] == title, ['t1', 't2', 'y', 'sim']]

            # Is this entity actually seen (=1) or unseen (=0)?
            actual = np.sum(df['y'].to_numpy())
            num_actual_seen += actual

            similarities = torch.Tensor(df['sim'].to_numpy() / 100).to(self._device)

            predicted_label = None
            # print(df)

            # If no similar candidate clusters have been found in the Query Set, the title does not match any of
            # the existing entities. The algorithm calls add_to_new_cluster.
            if df.shape[0] == 0:
                if actual == 0:
                    num_correct_unseen += 1
                else:
                    num_incorrect_unseen += 1

                # print(f"==== No label was assigned\n")
                unmatched_entities.append((title, ground_truth_label))

            else:
                # Query the model. Identify whether this title matches any of the existing candidate entities.
                eval_dataloader = eval_dataset.create_data_loader(df, sort_col=None, label_col='y')
                # print("==== MATCHING TITLE: ", title, ": ", actual)
                with torch.no_grad():
                    for n_batch, batch in enumerate(eval_dataloader):
                        input_ids, attention_mask, batch_labels = [b.to(self._device) for b in batch]
                        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

                        if n_batch == 0:
                            output_logits = outputs.logits
                            labels = batch_labels
                        else:
                            output_logits = torch.cat((output_logits, outputs.logits))
                            labels = torch.cat((labels, batch_labels))

                # Add the vector similarity to the output logits
                output_logits[:, 1] += similarities

                # Determine the matching entities of this title.
                # There are 3 possible outcomes: i) no matching entity was found, ii) one matching entity was found,
                # and iii) multiple matching entity were found. The algorithm handles each case individually.
                predictions = torch.argmax(output_logits, dim=-1)
                num_predicted_labels = predictions.sum().item()

                # print("\tReal Labels for Batch", n_batch, ":", labels, "\tModel Predictions:", predictions)
                # print("\tOnes:", num_predicted_labels, "outputs:", output_logits)

                if num_predicted_labels > 0:
                    idx_of_predictions = torch.where(predictions == torch.tensor(1))[0].tolist()
                    # print("\tPossible matches\n\tIndexes:", idx_of_predictions,
                    #      "\n\tLabels:", df['t2'].iloc[idx_of_predictions])

                    if actual == 1:
                        num_correct_seen += 1
                    else:
                        num_incorrect_seen += 1

                    # The model found one matching entity for the given title.
                    # We trust the model's output and assign the title to that cluster.
                    if num_predicted_labels == 1:
                        idx_in_df = idx_of_predictions[0]

                    # The model outputs multiple ones, i.e. it thinks that the given title matches multiple entities.
                    # Assign the entity that has the greatest output logit (as it was provided by the classifier).
                    # Note: Max similarity between sentence embeddings does not work equally well.
                    else:
                        # Among all the logits that output 1, find the greatest one
                        max_logit = torch.max(output_logits[idx_of_predictions, 1])
                        max_logit_idx = torch.where(output_logits == max_logit)[0].cpu().detach().numpy()
                        # print(max_logit, "-", max_logit_idx)
                        idx_in_df = max_logit_idx[0]

                    predicted_label = df['t2'].iloc[idx_in_df]
                    num_correct_labeled += df['y'].iloc[idx_in_df]

                # The model found no matching entities for the given title. It will now try to find a matching entity
                # from the new_clusters. If this is also unsuccessful, a new cluster will be created.
                if predicted_label is None:
                    # if (df['y'] == 0).all():
                    if actual == 0:
                        num_correct_unseen += 1
                    else:
                        num_incorrect_unseen += 1

                    # print(f"==== No label was assigned\n")
                    unmatched_entities.append((title, ground_truth_label))

            # print(f"Correctly Seen: {num_correct_seen} - Incorrectly Seen: {num_incorrect_seen} - "
            #      f"Correctly Unseen: {num_correct_unseen} - Incorrectly Unseen: {num_incorrect_unseen} === "
            #      f"Correctly Classified: {num_correct_labeled}/{num_actual_seen}")

        self.handle_unmatched_entities(unmatched_entities)

        result_record = FADRELResult(name=self._method_name, f=int(self._method_name[-1]))
        result_record.cor_seen = num_correct_seen
        result_record.inc_seen = num_incorrect_seen
        result_record.cor_unseen = num_correct_unseen
        result_record.inc_unseen = num_incorrect_unseen
        result_record.cor_classified = num_correct_labeled

        result_record.record(filename="results/results.csv")

    def handle_unmatched_entities(self, unmatched_entities : list):
        """
        An input title may match one, or none of the existing entities. In the latter case, the algorithm maintains
        self.new_clusters to accommodate these titles.
        In this function the algorithm tries to find a matching entity from new_clusters. If this is unsuccessful,
        a new cluster will be created.

        Args:
            unmatched_entities (str) : The title of the record that we try to match.
        """

        print("\nMatching the remaining unmatched entities...", flush=True)
        # Build an inverted index on the titles of the unmatched entities.
        entity_inv_index = {}
        unmatched_entities_df = pd.DataFrame(unmatched_entities, columns=['unmat_entity', 'ground_truth_label'])
        # unique_entities = list(set(unmatched_entities))
        unique_entities, indices = np.unique(unmatched_entities_df.loc[:, 'unmat_entity'].to_numpy(), return_index=True)

        for idx in indices:
            unique_title = unmatched_entities[idx][0]
            words = unique_title.split()
            for word in words:
                if word not in entity_inv_index:
                    entity_inv_index[word] = [idx]
                else:
                    entity_inv_index[word].append(idx)

        # Use the index to find the most suitable entities to search for
        num_unmatched_entities = len(unmatched_entities)
        unmatched_entities_pairs = []
        for idx in range(num_unmatched_entities):
            unmatched_entity_title = unmatched_entities[idx][0]
            unmatched_entity_label = unmatched_entities[idx][1]

            candidate_entity_ids = []
            for w in unmatched_entity_title.split():
                if w in entity_inv_index:
                    inverted_list = entity_inv_index[w]
                    candidate_entity_ids.extend(inverted_list)

            candidate_entity_ids = np.unique(candidate_entity_ids).tolist()
            candidate_entities = [unmatched_entities[idx] for idx in candidate_entity_ids]
            for candidate_entity in candidate_entities:
                match = 0
                if candidate_entity[1] == unmatched_entity_label:
                    match = 1
                unmatched_entities_pairs.append((unmatched_entity_title, candidate_entity[0], match))

        # Now create a dataset with the candidate entities and ask the model whether these entities match each other
        unmatched_entities_df = pd.DataFrame(unmatched_entities_pairs, columns=['t1', 't2', 'y'])
        # unmatched_entities_df.to_csv("unmatched_entities.csv", index=False)

        new_cluster_dataset = SentencePairDataset(self._tokenizer, self._batch_size, self._max_length)
        dataloader = new_cluster_dataset.create_data_loader(unmatched_entities_df, sort_col=None, label_col='y')
        self._model.eval()
        with torch.no_grad():
            for n_batch, batch in enumerate(dataloader):
                input_ids, attention_mask, batch_labels = [b.to(self._device) for b in batch]
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

                if n_batch == 0:
                    output_logits = outputs.logits
                else:
                    output_logits = torch.cat((output_logits, outputs.logits))

        # Prepare a dictionary (entity -> cluster) for fast entity assignment
        unmatched_entities_dict = {}
        for unmatched_entity in unmatched_entities:
            unmatched_entities_dict[unmatched_entity[0]] = 0

        # Assign entities to the records according to the model's predictions
        predictions = torch.argmax(output_logits, dim=-1).cpu().detach().numpy()
        print(predictions)
        num_new_clusters, ctr = 0, 0
        for pair in unmatched_entities_pairs:
            # print(pair[0], '-', pair[1], ':', predictions[ctr])
            if predictions[ctr] == 1:
                if unmatched_entities_dict[pair[0]] == 0 and unmatched_entities_dict[pair[1]] == 0:
                    num_new_clusters += 1
                    unmatched_entities_dict[pair[0]] = num_new_clusters
                    unmatched_entities_dict[pair[1]] = num_new_clusters
                elif unmatched_entities_dict[pair[0]] > 0:
                    unmatched_entities_dict[pair[1]] = unmatched_entities_dict[pair[0]]
                elif unmatched_entities_dict[pair[1]] > 0:
                    unmatched_entities_dict[pair[0]] = unmatched_entities_dict[pair[1]]
            ctr += 1
        print(unmatched_entities_dict)

        # print(unmatched_entities)
        # print(unmatched_entities_pairs)

