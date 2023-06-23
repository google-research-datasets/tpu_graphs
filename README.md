# TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

TpuGraphs is a performance prediction dataset on full tensor programs, represented as computational graphs, running on Tensor Processing Units (TPUs). Each graph in the dataset represents the main computation of a machine learning workload, e.g., a training epoch or an inference step. Each data sample contains a computational graph, a compilation configuration, and the execution time of the graph when compiled with the configuration. The graphs in the dataset are collected from open-source machine learning programs, featuring popular model architectures (e.g., ResNet, EfficientNet, Mask R-CNN, and Transformer). The dataset is located at `http://download.tensorflow.org/data/tpu_graphs/v0`.

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

To run the tiles baselines, you should download the tiles dataset:
```
# Create dataset directory
mkdir -p ~/data/tpugraphs
cd ~/data/tpugraphs
wget http://download.tensorflow.org/data/tpu_graphs/v0/LICENSE > LICENSE

# Copy data from to local dir.
cd ~/data/tpugraphs
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_tiles_tile_train.tar > npz_tiles_tile_train.tar
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_tiles_tile_valid.tar > npz_tiles_tile_valid.tar
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_tiles_tile_test.tar > npz_tiles_tile_test.tar
tar xvf npz_tiles_tile_train.tar
tar xvf npz_tiles_tile_valid.tar
tar xvf npz_tiles_tile_test.tar
```

#### Train model

The following command will train a GraphSAGE model with the early join of config features on a small subset of data:
```
python tiles_train.py --model=EarlyJoinSAGE --toy_data=True
```

To train on the full dataset, run:
```
python tiles_train.py --model=EarlyJoinSAGE
```

Once the training is done, it will produce a jsonz file with the prefix "run_".
This file will contain the overall top-K errors on kernels in the validation set.
To view the result:
```
zcat run_xxx.jsonz > run_xxx.json
```

Search for:
```
"final_error": {"val": {"1": <top-1 error>, "5": <top-5 error>, "10": <top-10 error>}}
```
where 0.2 error means 20% error.

Please refer to
[train_args.py](https://github.com/google-research-datasets/tpu_graphs/blob/main/baselines/tiles/train_args.py)
for a list of flags.

#### Sweep hyperparameters

Run Apache Beam locally (for debugging):
```
python tiles_beam_experiments.py --debug
```

To run the pipeline on Google Cloud, please follow [this instruction](https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python).


#### Evaluate model
Once the training is done, the training output directory specified with
`--out_dir` (~/out/tpugraphs_tiles by default) will contain a model directory,
whose name starts with the prefix `model_`.

To evaluate a model(s), run:
```
python tiles_evaluate.py --dirs <comma-separated list of model dirs>
```

This script will print out per-program top-K errors for kernels in the validation set in the following format:
```
{
  "K": {  # top-K error
    <program> : <error>,
    ...
  },
  ...
}
```

### Layout Model

Instructions to train the baseline models for the layout collection can be found at https://github.com/kaidic/GST.

