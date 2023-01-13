# [Moreh] Running on HAC VM - Moreh AI Framework


## Prepare

### Data
For testing purpose, we use `imagenet_100cls`, a subset of ImageNet with 100 classes.
Get the dataset from [here](http://ref.deploy.kt-epc.moreh.io:8080/reference/dataset/imagenet_100cls.tar.gz)
and extract it. The structure of the dataset is already compatible.

### Code
```bash
git clone https://github.com/loctxmoreh/Swin-Transformer
cd Swin-Transformer
```

### Environment
Although the original repo requires `torch==1.8.0`, `torch==1.7.1` works fine.
We're going to use `torch==1.7.1`

#### On A100 machine
```bash
conda env create -f a100env.yml
conda activate swin
```

#### On HAC machine
Edit `hacenv.yml`, pick a `torch` version (1.7.1 or 1.10.0) and uncomment the other.
Then:
```bash
conda env create -f hacenv.yml
conda activate swin
update-moreh --force --target 23.1.1            # run this command if you pick 1.7.1,
update-moreh --force --target 23.1.1 --nightly  # or this if you pick 1.10.0
```

#### *[Optional]* Fused window process
Optionally, install fused window process for acceleration:
```bash
cd kernels/window_process
python3 setup.py install #--user
```

Test again to make sure fused window process installed successfully:
```bash
python3 unit_test.py
```

Note that fused window process can only be installed on A100 VM. Installation
on HAC VM raises the following error:
`ValueError: Unknown CUDA arch (0.0) or GPU not supported`

After successfully installing this, the `--fused_window_process` flag can be
passed to the training program to accelerate.


## Run
Before running, edit `config.py` and set `_C.TRAIN.EPOCHS` to the desired number.
Note that, since warmup for Cosine Learning Rate Scheduler is enabled by default, the
number of training epochs should be at least 25.

To train from scratch, edit and run the `a100-train-from-scratch` script on
A100 VM and `hac-train-from-scratch` script on HAC VM:
- `--data-path`: point to the `imagenet_100cls` directory
- `--cfg`: point to config file in `./configs/swin/`, depending on which model configuration we want to train
- `--master_port`: point to any free port on the machine
- `--batch-size`: training batch size for a single GPU
- `--fused_window_process`: use the fused window process (if installed successfully)

### Note on HAC VM
The `torch.distributed` seems to not behave correctly on HAC VM. Because of
this, we have to manually set some env vars, as in `hac-train-from-scratch`.
Though, ultimately this is just a *hack*. Please do not expect this to work
consistently under the context of distributed training.
