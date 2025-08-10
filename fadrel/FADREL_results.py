class FADRELResult:
    def __init__(self, name, f):
        self.method_name = name
        self.fold = f

        self.cor_seen = 0
        self.inc_seen = 0
        self.cor_unseen = 0
        self.inc_unseen = 0
        self.cor_classified = 0

        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def display(self):
        print(f"Correctly Seen: {self.cor_seen} - Incorrectly Seen: {self.inc_seen} - "
              f"Correctly Unseen: {self.cor_unseen} - Incorrectly Unseen: {self.inc_unseen} === "
              f"Correctly Classified: {self.cor_classified}")

        print("Seen/Unseen classification performance:")
        print(f"Accuracy: {self.accuracy: .4f} - Precision: {self.precision: .4f} - Recall: {self.recall: .4f}"
              f"- F1: {self.f1: .4f}")

    def record(self, filename):

        self.accuracy = ((self.cor_seen + self.cor_unseen) /
                    (self.cor_seen + self.cor_unseen + self.inc_seen + self.inc_unseen))
        self.precision = self.cor_seen / (self.cor_seen + self.inc_seen)
        self.recall = self.cor_seen / (self.cor_seen + self.inc_unseen)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        self.display()

        with open(filename, "a+") as f:
            f.write(
                str(self.method_name) + "," + str(self.cor_seen) + "," + str(self.inc_seen) + "," +
                str(self.cor_unseen) + "," + str(self.inc_unseen) + "," +
                str(self.cor_classified) + "," + str(self.accuracy) + "," +
                str(self.precision) + "," + str(self.recall) + "," + str(self.f1) + "\n"
            )
