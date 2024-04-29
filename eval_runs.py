from tbparse import SummaryReader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class MySummaryReader:

    def __init__(self, log_dir):
        self.reader = SummaryReader(log_dir)

    def list_metrics(self):
        return self.reader.scalars['tag'].unique()
    
    def get_metric(self, metric_name):
        data = self.reader.scalars[self.reader.scalars['tag'] == metric_name].sort_values(by='step', ascending=True)[['step', 'value']].values
        return data[:, 0].astype(int), data[:, 1] # steps, values
    


if __name__ == "__main__":

    base_dir = Path("tb_logs")

    train_full_key = "val_acc_full_training/dataloader_idx_1"
    val_full_key = "val_acc_val_set/dataloader_idx_0"

    train_accs = []
    val_accs = []

    for i in range(6):
        my_reader = MySummaryReader(base_dir / f"DynamicCapacityCNN_{i}")

        steps, values = my_reader.get_metric(val_full_key)

        # Get value and index of maximum acc
        max_idx = np.argmax(values)
        max_acc = values[max_idx]

        # Get the corresponding train acc
        _, train_values = my_reader.get_metric(train_full_key)
        train_acc = train_values[max_idx]

        # Append to lists
        train_accs.append(train_acc)
        val_accs.append(max_acc)

        #print("Max accuracy: ", max_acc)

    # Plot the results
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("MEC")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()