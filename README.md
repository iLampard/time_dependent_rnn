# time_dependent_rnn

In order to practise Tensorflow, I implement the model of [Time Dependent Representation of Neural Event Sequence Prediction](https://arxiv.org/abs/1708.00065)
based on my own understanding.


## How to run

### Download sample data
Download three pickles from [Baidu Disk](https://pan.baidu.com/s/1O5yLWckrWzFaS9ku2N_7xw) with password *fn40* and 
put them under *tdrnn/data/data_so*.

```bash
tdrnn
 |__ data
       |__ data_so
            |__ dev.pkl
            |__ test.pkl
            |__ train.pkl
```

Please be noted that these data is preprocessed by my own and not exactly the same one used in the paper. 


### Run the training and prediction pipeline

Suppose we want to run 100 epochs and use Tensorboard to visualize the process

```bash
cd tdrnn
python main.py --write_summary True --max_epoch 200
```

To check the description of all flags
```bash
python main.py -helpful
```

To open tensorboard
```bash
tensorboard --logdir=path
```
where the path can be found in the log which shows the relative dir to save the model, e.g. *logs/data_so/ModelWrapper/lr-0.01_dim-64_drop-0.0/20200201-011244
/saved_model/tfb_dir/*.

and then open the browser

<img src="https://github.com/iLampard/time_dependent_rnn/tree/master/figures/loss.png"/>

<img src="https://github.com/iLampard/time_dependent_rnn/tree/master/figures/acc.png"/>


## Requirement

```bash
tensorflow==1.13.1
```