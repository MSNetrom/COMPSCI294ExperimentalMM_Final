# COMPSCI294ExperimentalMM_Final

I choose to use **Lightning** to make implementation easier and more readable.
All of this is logged to tensorboard. Best models are saved to disk, measured by the validation loss (from normal validation set).

## Setup

Make sure you have packages in requirements.txt installed. You can install them by running:

```python3 -m pip install -r requirements.txt```

## Train

To train the model, run:

```python3 main.py```

Edit main.py to fit your use.

## Base model

Base model is found in ```base_cnn.py```. This model could be inherited from for constructing other designs.

## Tensorboard

To visualize the training process, run:

```tensorboard --logdir=tb_logs/my_model/version_<version_number>```

## Evaluation

Some predictions about the design can be found by running ```design_finder.py``` and ```calcs.ipynb```. 
Evaluation of trained models can be done by running ```eval_runs.py```.

-------------------

Best,
Morten Svendgård