## Setting up the environment
> Conda environment highly preferred over python venv, since some packages are available in conda only. Also for better dependency management.
- With the available conda `environment.yml` file, run the below command:
```bash
conda env create -f environment.yml
```
- This will create a new conda environment named `infragan` with all the required packages.
- There might have issues with CUDA toolkit versions, if latest isn't supported in your device. Check versions and reinstall the relevant pytorch version accordingly.

## Commands usage

- Multi-GPU processing can be attempted by passing in `--gpu_ids 0,1` etc. instead of just `--gpu_ids 0`.

<div style="page-break-after: always;"></div>

### For 256 resolution training

#### ThermalGAN VEDAI
```bash
python train.py --dataset_mode VEDAI --dataroot ./datasets/VEDAI_sample --name thermal_gan_vedai --model thermal_gan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 1 --save_epoch_freq 1 --resolution 256
```

#### InfraGAN VEDAI
```bash
python train.py --dataset_mode VEDAI --dataroot ./datasets/VEDAI_sample --name infragan_vedai --model infragan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 1 --save_epoch_freq 1 --resolution 256
```

#### InfraGAN KAIST
```bash
python train.py --dataset_mode KAIST_new --dataroot ./datasets/KAIST_sample --name infragan_kaist --model infragan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 1 --save_epoch_freq 1 --resolution 256
```

#### InfraGAN FLIR
```bash
python train.py --dataset_mode FLIR_new --dataroot ./datasets/FLIR_sample --name infragan_flir --model infragan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 1 --save_epoch_freq 1 --resolution 256
```

<div style="page-break-after: always;"></div>

### For 256 resolution testing
> Note: Need to specify total number of images being processed in the `--how_many` flag, default is 10000 upper limit

#### ThermalGAN VEDAI
```bash
python test.py --dataset_mode VEDAI --dataroot ./datasets/VEDAI_sample --name thermal_gan_vedai --model thermal_gan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --norm batch --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 4 --how_many 5 --resolution 256
```

#### InfraGAN VEDAI
```bash
python test.py --dataset_mode VEDAI --dataroot ./datasets/VEDAI_sample --name infragan_vedai --model infragan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --norm batch --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 4 --how_many 5 --resolution 256
```

#### InfraGAN KAIST
```bash
python test.py --dataset_mode KAIST_new --dataroot ./datasets/KAIST_sample --name infragan_kaist --model infragan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --norm batch --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 4 --how_many 5 --resolution 256
```

#### InfraGAN FLIR
```bash
python test.py --dataset_mode FLIR_new --dataroot ./datasets/FLIR_sample --name infragan_flir --model infragan --which_model_netG unet_256 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --norm batch --loadSize 256 --fineSize 256 --gpu_ids 0 --nThreads 8 --batchSize 4 --how_many 5 --resolution 256
```

<div style="page-break-after: always;"></div>

## Required Dataset Structure

### `VEDAI` dataset mode
```
dataset_directory/
├── train/
│   ├── 00001000_co.png
│   ├── 00001000_ir.png
│   ├── 00001001_co.png
│   ├── 00001001_ir.png
│   └── ...
├── test/
│   ├── 00001002_co.png
│   ├── 00001002_ir.png
│   ├── 00001003_co.png
│   ├── 00001003_ir.png
│   └── ...
```

### `KAIST_new` dataset mode

```
dataset_directory/
├── train/
│   ├── lwir/
│   │   ├── set00_V000_I01231_1.jpg
│   │   └── set00_V000_I01231_2.jpg
│   │   └── ...
│   ├── visible/
│   │   ├── set00_V000_I01231_1.jpg
│   │   └── set00_V000_I01231_2.jpg
│   │   └── ...
├── test/
│   ├── lwir/
│   │   ├── set00_V000_I01232_1.jpg
│   │   └── set00_V000_I01232_2.jpg
│   │   └── ...
│   ├── visible/
│   │   ├── set00_V000_I01232_1.jpg
│   │   └── set00_V000_I01232_2.jpg
│   │   └── ...
```

<div style="page-break-after: always;"></div>

## `FLIR_new` dataset mode
```
dataset_directory/
├── train/
│   ├── JPEGImages/
│   │   ├── FLIR_00002_PreviewData_0.jpeg
│   │   ├── FLIR_00002_RGB_0.jpg
│   │   └── ...
├── test/
│   ├── JPEGImages/
│   │   ├── FLIR_00003_PreviewData_0.jpeg
│   │   ├── FLIR_00003_RGB_0.jpg
│   │   └── ...
```


## Changes made to the original python scripts

### For Bug fixing:

#### `models/networks.py`
```python
if isinstance(input, tuple):
    input = input[0]
```
- Added the above line to `networks.py` at line 190 and line 212.
- This was quickfix for the error of tuple being returned instead of tensor. Error shows up when running `train.py`.

#### `test.py`
- line 49:
```python
from tdqm import tqdm
for i, data in tqdm(enumerate(dataloader)):
```
- Crucial fix for the `Expected 4D tensor (got 3D tensor)` error. Earlier it was iterating over dataset instead of dataloader. The dataloader provides the additional 4th dimension of batch size.
- Added tqdm for some more convenience features like estimating how long it could run.

#### `util/visualizer.py`
- line 122:
```python
if total_iter == 0: total_iter = 1e-10
```
- Modified in an attempt to fix ZeroDivisionError. (Why it was allowed to happen in the first place when `total_iter` obviously would be 0 at some point, I still have no clue.)

### New features added

#### `options/base_options.py`
- line 45:
```python
self.parser.add_argument('--resolution', type=int, default=512, help='Resolution of the Discriminator')
```
- Earlier within the code it would always assume to be 512, added this option to allow for 256 resolution images as well.

#### `models/thermal_gan_model.py`
- line 24:
```python
res = opt.resolution
self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, resolution=res)
```
- To allow for custom resolution for the discriminator for ThermalGAN.

#### `models/infragan.py`
- line 24:
```python
res=256 if opt.dataset_mode == 'FLIR' else opt.resolution
self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, resolution=res)
```
- Same as earlier, to allow for custom resolution for the discriminator for InfraGAN.

#### `data/thermal_dataset.py` - for `KAIST_new` and `FLIR_new` dataset modes
- Made two new functions, `make_thermal_dataset_kaist_sanitized(path)` and `make_thermal_dataset_flir_aligned(path)`.

#### For supporting `KAIST_new` and `FLIR_new` dataset modes
- `data/custom_dataset_data_loader.py` line 16
- `eval.py` line 9
- `test.py` line 22

#### `options/test_options.py` (optional)
- Line 12: changed default value of --how_many from 50 to 10000.

#### `util.py` (optional)
- Commented out the `tmin`, `tmax` printing coz it was annoying.


## `.vscode/` configuration files
- I have set up `launch.json`, `task.json` files for convenient use of VS Code python debugger, and automating running the visdom server.
