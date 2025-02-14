# REEF: Relations Are What Graph Foundation Models Need

## Environment Settings

Core Dependencies:
- torch==1.13.1+cu117
- transformers==4.43.2
- sentence-transformers==3.1.1
- python==3.10.14

PyG-related Dependencies:
- torch-cluster==1.6.0+pt113cu117
- torch-geometric==2.6.1
- torch-scatter==2.1.0+pt113cu117
- torch-sparse==0.6.15+pt113cu117
- torch-spline-conv==1.2.1+pt113cu117
- pyg-lib==0.4.0+pt113cu117

For further details, please refer to the `requirements.txt` file.

## Folder & File Description

- `code/`: Contains all the code for the project.
- `data/`: Contains all the raw data.
- `cache_data/`: Contains the cached data for the constructed subgraphs.
- `pretrained_data/`: Contains pretrained data.
- `pretrained_result/`: Contains the pretrained model parameters.
- `code/downstream.py`: Contains code for downstream tasks.
- `code/finetune.py`: Contains code for fine-tuning.
- `code/main.py`: Contains code for pretraining.
- `code/utils.py`: Contains utility functions.
- `code/args.py`: Contains the hyperparameters and settings for the project.

## Usage

### Pre-training

```bash
python main.py
```

### Finetuning

```bash
python finetune.py
```

