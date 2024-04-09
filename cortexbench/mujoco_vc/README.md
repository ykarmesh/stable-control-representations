# MetaWorld

We test the pretrained visual reprensetations in the few-shot visual imitation learning setting. The experiment design for this is based on the [PVR paper](https://sites.google.com/view/pvr-control).


## Downloading demonstration datasets
To download the demonstration datasets, create a directory for the dataset by executing the following command:
```bash
mkdir -p benchmarks/mujoco_vc/visual_imitation/data/datasets
cd benchmarks/mujoco_vc/visual_imitation/data/datasets
```

Then, download the dataset with the following commands:
### Metaworld benchmark:
```bash
wget https://dl.fbaipublicfiles.com/eai-vc/mujoco_vil_datasets/metaworld-expert-v1.0.zip
unzip metaworld-expert-v1.0.zip
rm metaworld-expert-v1.0.zip
```

## Running experiments
To run experiments, navigate to the `visual_imitation` subdirectory, which contains launch scripts and config files, by executing the following commands:
```bash
cd stable-control-representations/benchmarks/mujoco_vc/visual_imitation/
```
To run the job, execute the following script:
```bash
./run_script.sh
```