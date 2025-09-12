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
                 text_vectorizer : SentenceTransformer, method_name : str, evaluate : bool) -> None:
        """
        Initialize a MatchMaker object.

        Args:
            classifier (SentencePairClassifier): The BERT Classification model to be used for matching.
            batch_size (int): The batch size to be used during testing.
            max_length (int): The maximum embedding length of the BERT Classification model.
            method_name (str): The name of the entity matching method to be evaluated.
            evaluate (bool): Whether to evaluate the matching performance (requires ground truth entities).
        """
        self._model = classifier.model
        self._tokenizer = classifier.tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._method_name = method_name
        self._eval = evaluate
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_new_clusters = 0
        self.new_clusters = {}
        self.text_vectorizer = text_vectorizer

    def perform_matching(self, query_df:pd.DataFrame, test_df : pd.DataFrame, entities : dict,
                         title_col : str, entity_label_col : str, entity_id_col : str) -> None:
        """
        Find the matching entities for the unknown input record titles. An input title may match one, or none of
        the existing entities. In the latter case, the algorithm maintains self.new_clusters to accommodate these
        titles.

        Args:
            query_df (pd.DataFrame) : A DataFrame that contains the candidate entities for each record title.
            test_df (pd.DataFrame) : A DataFrame that contains the unknown (unmatched) records
            entities (dict) : A dictionary of Entity objects
            title_col (str) : The column name of test_df that contains the titles of the unmatched records.
            entity_label_col (str) : The column name of test_df that contains the ground truth entity labels of the records.
            entity_id_col (str) : The column name of test_df that contains the ground truth entity IDs of the records.
        """
        print("\nMatching the unlabeled entities...", flush=True)
        eval_dataset = SentencePairDataset(self._tokenizer, self._batch_size, self._max_length)

        # This list stores the records that do not match with any of the existing entities.
        umat_records = []

        self._model.eval()

        # For each record title in the query set, retrieve the candidate entity labels and create a dataloader.
        # We will compare each title with all candidate (i.e. most similar) entity labels.
        test_records_df = test_df.loc[:, [title_col, entity_label_col, entity_id_col]]

        num_cor_labeled = 0
        num_cor_seen, num_incor_seen, num_cor_unseen, num_incor_unseen, num_actual_seen = 0, 0, 0, 0, 0

        # print(test_df.shape)
        for n_row, row in tqdm(enumerate(test_records_df.itertuples())):
            title = row[1]
            ground_truth_label = row[2]
            ground_truth_entity_id = row[3]

            # Fetch the candidate cluster labels from the query set. The query set has been built in FADREL_match.py
            # and contains the labels that have at least one word in common with the entity title.
            df = query_df.loc[query_df['t1'] == title, ['t1', 't2', 'y', 'sim']]

            # Is this entity actually seen (=1) or unseen (=0)?
            actual = 0
            if ground_truth_entity_id in entities:
                actual = 1

            num_actual_seen += actual

            similarities = torch.Tensor(df['sim'].to_numpy() / 100).to(self._device)

            # If no similar candidate entities have been found in the Query Set, the title does not match any of
            # the existing entities. We append this title to umat_records.
            predicted_label = None
            if df.shape[0] == 0:
                if actual == 0:
                    num_cor_unseen += 1
                else:
                    num_incor_unseen += 1

                # print(f"==== No label was assigned\n")
                umat_records.append((title, ground_truth_label))

            else:
                # Query the model. Identify whether this title matches any of the candidate entities.
                eval_dataloader = eval_dataset.create_data_loader(df, sort_col=None, label_col='y')
                # print("==== MATCHING RECORD TITLE: ", title, ": ", actual)
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
                # and iii) multiple matching entities were found. The algorithm handles each case individually.
                predictions = torch.argmax(output_logits, dim=-1)
                num_predicted_labels = predictions.sum().item()

                # print("\tReal Labels for Batch", n_batch, ":", labels, "\tModel Predictions:", predictions)
                # print("\tOnes:", num_predicted_labels, "outputs:", output_logits)

                if num_predicted_labels > 0:
                    idx_of_predictions = torch.where(predictions == torch.tensor(1))[0].tolist()
                    # print("\tPossible matches\n\tIndexes:", idx_of_predictions,
                    #      "\n\tLabels:", df['t2'].iloc[idx_of_predictions])

                    if actual == 1:
                        num_cor_seen += 1
                    else:
                        num_incor_seen += 1

                    # The model found one matching entity for the given title.
                    # We accept the model's output, and we consider that the title matches this entity.
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
                    num_cor_labeled += df['y'].iloc[idx_in_df]

                # The model found no matching entities for the given title. We append this title to umat_records.
                if predicted_label is None:
                    # if (df['y'] == 0).all():
                    if actual == 0:
                        num_cor_unseen += 1
                    else:
                        num_incor_unseen += 1

                    # print(f"==== No label was assigned\n")
                    umat_records.append((title, ground_truth_label))

            # print(f"corly Seen: {num_cor_seen} - Incorrectly Seen: {num_incor_seen} - "
            #      f"corly Unseen: {num_cor_unseen} - Incorrectly Unseen: {num_incor_unseen} === "
            #      f"corly Classified: {num_cor_labeled}/{num_actual_seen}")

        # Now process all the records for which no matching entity was determined.
        # self.handle_unmatched_records(umat_records)

        # Record the evaluation results.
        result_record = FADRELResult(name=self._method_name, f=int(self._method_name[-1]))
        result_record.cor_seen = num_cor_seen
        result_record.inc_seen = num_incor_seen
        result_record.cor_unseen = num_cor_unseen
        result_record.inc_unseen = num_incor_unseen
        result_record.cor_classified = num_cor_labeled
        result_record.record(filename="results/results.csv")

    def handle_unmatched_records(self, umat_records : list):
        """
        An input title may match one, or none of the existing entities. In the latter case, the algorithm maintains
        self.new_clusters to accommodate these titles.
        In this function the algorithm tries to find a matching entity from new_clusters. If this is unsuccessful,
        a new cluster will be created.

        Args:
            umat_records (str) : The title of the record that we try to match.
        """

        print("\nMatching the remaining unmatched entities...", flush=True)

        np_umat_records = np.array(umat_records)
        print("Number of entities for unmatched records:", np.unique(np_umat_records[:, 1]).size)

        # Build an inverted index on the titles of the unmatched records.
        records_inv_index = {}
        umat_records_df = pd.DataFrame(umat_records, columns=['unmat_title', 'ground_truth_label'])
        unique_entities, indices = np.unique(umat_records_df.loc[:, 'unmat_title'].to_numpy(), return_index=True)

        for idx in indices:
            unique_title = umat_records[idx][0]
            words = unique_title.split()
            for word in words:
                if word not in records_inv_index:
                    records_inv_index[word] = [idx]
                else:
                    records_inv_index[word].append(idx)

        # Use the index to find the most suitable entities to search for
        num_unmatched_records = len(umat_records)
        print("Number of unmatched entities: ", num_unmatched_records)

        umat_title_pairs = []
        for idx in range(num_unmatched_records):
            umat_record_title = umat_records[idx][0]
            ground_truth_label = umat_records[idx][1]

            candidate_record_ids = []
            for w in umat_record_title.split():
                if w in records_inv_index:
                    inverted_list = records_inv_index[w]
                    candidate_record_ids.extend(inverted_list)

            candidate_record_ids = np.unique(candidate_record_ids).tolist()
            candidate_records = [umat_records[idx] for idx in candidate_record_ids]
            for candidate_record in candidate_records:
                match = 0
                if candidate_record[1] == ground_truth_label:
                    match = 1
                umat_title_pairs.append((umat_record_title, candidate_record[0], match))

        # Now create a dataset with the candidate records and ask the model whether these record titles match
        umat_records_df = pd.DataFrame(umat_title_pairs, columns=['t1', 't2', 'y'])
        # unmatched_records_df.to_csv("unmatched_entities.csv", index=False)

        umat_dataset = SentencePairDataset(self._tokenizer, self._batch_size, self._max_length)
        dataloader = umat_dataset.create_data_loader(umat_records_df, sort_col=None, label_col='y')
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
        umat_records_dict = {}
        for umat_record in umat_records:
            umat_records_dict[umat_record[0]] = 0

        # Assign entities to the records according to the model's predictions
        predictions = torch.argmax(output_logits, dim=-1).cpu().detach().numpy()
        print(predictions)
        num_new_entities, ctr = 0, 0
        for pair in umat_title_pairs:
            # print(pair[0], '-', pair[1], ':', predictions[ctr])
            if predictions[ctr] == 1:
                if umat_records_dict[pair[0]] == 0 and umat_records_dict[pair[1]] == 0:
                    num_new_entities += 1
                    umat_records_dict[pair[0]] = num_new_entities
                    umat_records_dict[pair[1]] = num_new_entities
                elif umat_records_dict[pair[0]] > 0:
                    umat_records_dict[pair[1]] = umat_records_dict[pair[0]]
                elif umat_records_dict[pair[1]] > 0:
                    umat_records_dict[pair[0]] = umat_records_dict[pair[1]]
            ctr += 1
        print(umat_records_dict)

        # print(unmatched_entities)
        # print(unmatched_entities_pairs)

