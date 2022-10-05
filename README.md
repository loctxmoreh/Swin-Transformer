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
First, create conda environment:
```bash
conda create -n swin python=3.8
conda activate swin
```
Although the original repo requires `torch==1.8.0`, `torch==1.7.1` works fine.
We're going to install `torch==1.7.1`

#### `torch` on A100 VM
With `torch` version 1.7.1:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

With `torch` version 1.12.1:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

#### `torch` on HAC VM
```bash
conda install -y torchvision torchaudio numpy protobuf==3.13.0 pytorch==13.7.1 cpuonly -c pytorch
```
Then force update Moreh (version `22.9.2` at the moment this document is written)
```bash
update-moreh --force --target 22.9.2
```
#### The rest of the requirements
```bash
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

Optionally, install fused window process for acceleration:
```bash
cd kernels/window_process
python setup.py install #--user
```

Test again to make sure fused window process installed successfully:
```bash
python unit_test.py
```

Note that fused window process can only be installed on A100 VM. Installation
on HAC VM raises the following error:
`ValueError: Unknown CUDA arch (0.0) or GPU not supported`

Also note that on A100 VM, the CUDA version on the machine must **exactly
match** with the CUDA version that `torch` is compiled with. For example, if
`torch==1.7.1+cu110` then we must use CUDA 11.0 (by set `CUDA_HOME` to CUDA
11.0 directory) to install fused window process. If they do not match, the
installation log will only have some warnings, but `unit_test.py` will raise
the following error:
```
ImportError: /data/work/anaconda3/envs/swin2/lib/python3.8/site-packages/swin_window_process-0.0.0-py3.8-linux-x86_64.egg/swin_window_process.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN6caffe28TypeMeta21_typeMetaDataInstanceIdEEPKNS_6detail12TypeMetaDataEv
```

After successfully installing this, the `--fused_window_process` flag can be
passed to the training program to accelerate.


## Run
For testing purpose, edit `config.py` and set `_C.TRAIN.EPOCHS=1` or any desired
number of training epochs.

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
