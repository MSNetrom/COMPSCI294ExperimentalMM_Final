# COMPSCI294ExperimentalMM_Final

I choose to use **Lightning** to make implementation easier and more readable. 

This code splits train data into train and validation data. It then trains a simple CNN model on the training data and validates during training. It not only validates on the validation data, but also the whole training data set every epoch. At the end of training, it tests the model on the test data.
All of this is logged to tensorboard. Best models are saved to disk, measured by the validation loss (from normal validation set).

## Setup

Make sure you have packages in requirements.txt installed. You can install them by running:

```python3 -m pip install -r requirements.txt```

## Train

To train the model, run:

```python3 main.py```

## Tensorboard

To visualize the training process, run:

```tensorboard --logdir=tb_logs/my_model/version_<version_number>```

## Edit model

I have tried to make some good code for us to start with. It might be overwhleming at first, but I think it will be helpful in the long run.
Really everything needed to actually change the architecture of the model is to edit or make something similar to ```OurCNN``` in ```model.py```.
```OurCNN``` is a simple CNN model that I made to get us started.

-------------------

Best,
Morten Svendgård