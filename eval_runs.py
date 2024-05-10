from tbparse import SummaryReader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from strategy2_networks import STRAT1_TEST_LIST

class MySummaryReader:

    def __init__(self, log_dir: Path):
        self.reader = SummaryReader(log_dir)

    def list_metrics(self):
        return self.reader.scalars['tag'].unique()
    
    def get_metric(self, metric_name):
        data = self.reader.scalars[self.reader.scalars['tag'] == metric_name].sort_values(by='step', ascending=True)[['step', 'value']].values
        return data[:, 0].astype(int), data[:, 1] # steps, values
    

def read_strategy1(name_addon="_Chinese") -> List[Tuple[int, MySummaryReader]]:
    # We want to return a list of (mec, readers)

    hidden_sizes = [4, 6, 9, 13, 19, 56, 222, 888, 3552]

    result_list = []

    for h in hidden_sizes:

        log_dir = Path("tb_logs") / f"Strat1_{h}{name_addon}"
        my_reader = MySummaryReader(log_dir)

        result_list.append((11*h, my_reader))

    return result_list

def read_strategy2(name_addon="_Chinese") -> List[Tuple[int, MySummaryReader]]:
    # We want to return a list of (mec, readers)

    result_list = []

    for model_class in STRAT1_TEST_LIST:

        log_dir = Path("tb_logs") / f"Strat2_{model_class.__name__}{name_addon}"
        my_reader = MySummaryReader(log_dir)

        result_list.append((model_class(input_size=(1, 64, 64)).get_mec(), my_reader))

    return result_list


def get_acc_vs_mec(summary_list: List[Tuple[int, MySummaryReader]]):

    """
    Gets accuracy of on training and validation based on best validation accuracy on each MEC
    """

    train_full_key = "val_acc_full_training/dataloader_idx_1"
    val_full_key = "val_acc_val_set/dataloader_idx_0"

    # We are gonna get the max accuracy based on the validation set, and corresponding training accuracy

    mec_vals = []
    train_accs = []
    val_accs = []

    for mec, reader in summary_list:

        _, values = reader.get_metric(val_full_key)

        # Get value and index of maximum acc
        max_idx = np.argmax(values)
        max_acc = values[max_idx]

        # Get the corresponding train acc
        _, train_values = reader.get_metric(train_full_key)
        train_acc = train_values[max_idx]

        # Append to lists
        mec_vals.append(mec)
        train_accs.append(train_acc)
        val_accs.append(max_acc)

    mec_vals = np.array(mec_vals)
    train_accs = np.array(train_accs)
    val_accs = np.array(val_accs)

    return mec_vals, train_accs, val_accs



if __name__ == "__main__":

    #train_examples = 10000
    #val_examples = 10000

    strat1_summary_list = read_strategy1()
    strat2_summary_list = read_strategy2()

    #print(strat1_summary_list[0][1].list_metrics())

    strat1_mec_vals, strat1_train_accs, strat1_val_accs = get_acc_vs_mec(strat1_summary_list)
    strat2_mec_vals, strat2_train_accs, strat2_val_accs = get_acc_vs_mec(strat2_summary_list)

    # Plot accuracy
    plt.plot(strat1_mec_vals, strat1_train_accs, label="Strategy 1 Train Accuracy", marker='x', linestyle='-')
    plt.plot(strat1_mec_vals, strat1_val_accs, label="Strategy 1 Validation Accuracy", marker='x', linestyle='-')
    plt.plot(strat2_mec_vals, strat2_train_accs, label="Strategy 2 Train Accuracy", marker='x', linestyle='-')
    plt.plot(strat2_mec_vals, strat2_val_accs, label="Strategy 2 Validation Accuracy", marker='x', linestyle='-')
    plt.xlabel("MEC")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs MEC")
    plt.legend()
    plt.grid()
    plt.savefig("figures/acc_vs_mec.pdf", format="pdf")
    plt.show()

    plt.plot(strat1_mec_vals, strat1_train_accs-strat1_val_accs, label="Strategy 1 [Train - Validation Accuracy]", marker='x', linestyle='-')
    plt.plot(strat2_mec_vals, strat2_train_accs-strat2_val_accs, label="Strategy 2 [Train - Validation Accuracy]", marker='x', linestyle='-')
    plt.xlabel("MEC")
    plt.ylabel("Difference in Accuracy")
    plt.title("(Difference in Accuracy) vs MEC")
    plt.legend()
    plt.grid()
    plt.savefig("figures/diff_acc_vs_mec.pdf", format="pdf")
    plt.show()

    # Plot zoomed versions of this
    idx_zoom_strat1 = np.where(strat1_mec_vals < 4000)
    idx_zoom_strat2 = np.where(strat2_mec_vals < 4000)

    plt.plot(strat1_mec_vals[idx_zoom_strat1], strat1_train_accs[idx_zoom_strat1], label="Strategy 1 Train Accuracy", marker='x', linestyle='-')
    plt.plot(strat1_mec_vals[idx_zoom_strat1], strat1_val_accs[idx_zoom_strat1], label="Strategy 1 Validation Accuracy", marker='x', linestyle='-')
    plt.plot(strat2_mec_vals[idx_zoom_strat2], strat2_train_accs[idx_zoom_strat2], label="Strategy 2 Train Accuracy", marker='x', linestyle='-')
    plt.plot(strat2_mec_vals[idx_zoom_strat2], strat2_val_accs[idx_zoom_strat2], label="Strategy 2 Validation Accuracy", marker='x', linestyle='-')
    plt.xlabel("MEC")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs MEC (Zoomed)")
    plt.legend()
    plt.grid()
    plt.savefig("figures/acc_vs_mec_zoomed.pdf", format="pdf")
    plt.show()

    plt.plot(strat1_mec_vals[idx_zoom_strat1], strat1_train_accs[idx_zoom_strat1]-strat1_val_accs[idx_zoom_strat1], label="Strategy 1 [Train - Validation Accuracy]", marker='x', linestyle='-')
    plt.plot(strat2_mec_vals[idx_zoom_strat2], strat2_train_accs[idx_zoom_strat2]-strat2_val_accs[idx_zoom_strat2], label="Strategy 2 [Train - Validation Accuracy]", marker='x', linestyle='-')
    plt.xlabel("MEC")
    plt.ylabel("Difference in Accuracy")
    plt.title("(Difference in Accuracy) vs MEC (Zoomed)")
    plt.legend()
    plt.grid()
    plt.savefig("figures/diff_acc_vs_mec_zoomed.pdf", format="pdf")
    plt.show()


    