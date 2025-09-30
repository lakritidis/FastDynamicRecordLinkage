class Record:
    def __init__(self, rec_id, rec_title, entity_id, embedding):
        self.id = rec_id
        self.title = rec_title
        self.entity_id = entity_id
        self.embedding = embedding

    def display(self, show_embeddings : bool=False) -> None:
        if show_embeddings:
            print("Record ID:", self.id, ", Title:", self.title, ", Entity ID:", self.entity_id,
                  ", Embedding:", self.embedding, flush=True)
        else:
            print("Record ID:", self.id, ", Title:", self.title, ", Entity ID:", self.entity_id, flush=True)
