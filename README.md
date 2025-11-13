### Enviroment

Please refer to the Muffin project, https://github.com/library-testing/Muffin?


### Experiments

1. Configuration

Please vim main.py

```
if __name__ == '__main__':
    DATASET_NAME = 'mnist' # `cifar10`, `mnist`, `sinewave`
    CASE_NUM = 10
    GENERATE_MODE = 'seq'  # `seq`, `merging`, `dag`

    main(DATASET_NAME, CASE_NUM, GENERATE_MODE)
```

- `DATASET_NAME` indicates the name of dataset. `cifar10`, `mnist`, `sinewave` can be choosed.
- `CASE_NUM` represents the number of models.
- `GENERATE_MODE` indicates the method of generating models. There are threes choices, i.e. `seq`, `merging`, `dag`.

2. Preprocessing

- Dataset generation: please execute the following command.
```
cd /data/dataset
source activate lemon
python get_dataset.py
```

- Sqlite3 database: please execute the following command.

```
cd /data/data
source activate lemon
python clear_data.py
```

3. Start

```
source activate lemon
python main.py
```


### Tensorflow 

tensorflow 2.16.1
keres 3.10 

commands:
1. pip install tensorflow==2.16.1


### Pytorch

pytorch 2.2.0
keres 3.10

commands:
1. pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
2. pip install numpy==1.26


