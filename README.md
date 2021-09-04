Our project structure is as follows:
```
├── dataset
│   ├── dev.pt
│   ├── item_info.csv
│   ├── items_info.pt
│   ├── test.pt
│   ├── track1_testset.csv
│   ├── track2_testset.csv
│   ├── train.pt
│   └── trainset.csv
├── output
│   ├── best_checkpoint.pt
├── process.sh
├── src
└── train.sh

```

# Data Processing
Download the `trainset.csv`, `track1_testset.csv`, `track2_testset.csv` and `item_info.csv` into `dataset` and then
run the following shell to process the data to obtain the pickle data.
```python
sh process.sh
```

# Train
Run the following shell to train the DQN, and it will save `best_checkpoint.pt` in `output`. 

Note that our best checkpoint have been saved in `output`.
```
sh train.sh
```

# Eval
Run the following shell to obtain the evaluating result on the track2 dataset, and it will generate `test_result.json` in `output`.
```
sh test.sh
```

# Upload result
Run the following shell to obtain the `submission.csv` based on the above `test_result.json`.
```
sh upload.sh --pred_file output/test_result.json
```
