# TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

TpuGraphs is a performance prediction dataset on full tensor programs, represented as computational graphs, running on Tensor Processing Units (TPUs). Each graph in the dataset represents the main computation of a machine learning workload, e.g., a training epoch or an inference step. Each data sample contains a computational graph, a compilation configuration, and the execution time of the graph when compiled with the configuration. The graphs in the dataset are collected from open-source machine learning programs, featuring popular model architectures (e.g., ResNet, EfficientNet, Mask R-CNN, and Transformer).

*This is not an officially supported Google product.*


## Running Baseline Models

### Tile Size Model

#### Python environment setup with Conda
```
conda create -n tpugraphs python=3.10
conda activate tpugraphs

conda install -c conda-forge tensorflow=2.12
conda install -c conda-forge tqdm

pip install tensorflow_gnn==0.5.0
pip install tensorflow-ranking==0.5.2
conda clean --all
```

For subsequent runs, simply activate the same environment with `conda activate tpugraphs`.

#### Copy dataset files

```
# Create dataset directory
mkdir -p ~/data/tpugraphs_tiles

# Copy data from to local dir.
TODO
```

#### Train model
The following command will train a GraphSAGE model with the early join of config features on a small subset of data:
```
python train_tiles.py --model=EarlyJoinSAGE --toy_data=True
```

To train on the full dataset, run:
```
python train_tiles.py --model=EarlyJoinSAGE
```

Please refer to
[train_args.py](https://github.com/google-research-datasets/tpu_graphs/blob/main/baselines/tiles/train_args.py)
for a list of flags.

#### Evaluate model

TODO

#### Sweep hyperparameters

TODO

