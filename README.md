# time_dependent_rnn

In order to practise Tensorflow, I implement the model of [Time Dependent Representation of Neural Event Sequence Prediction](https://arxiv.org/abs/1708.00065)
based on my own understanding.


## How to run

### Download sample data
I preprocessed StackOverflow data and save them in three pickles. Download them from [Baidu Disk](https://pan.baidu.com/s/1O5yLWckrWzFaS9ku2N_7xw) with password *fn40* and 
put them under *tdrnn/data/data_so*.

```bash
tdrnn
 |__ data
       |__ data_so
            |__ dev.pkl
            |__ test.pkl
            |__ train.pkl
```

### Run the main file

```bash
python main.py --max_epoch 100
```

If you want to activate tensorboard


```bash
python main.py --use_summary True
```


## Requirement

```bash
tensorflow==1.13.1
```