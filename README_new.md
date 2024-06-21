## Applying the patch file changes
- After cloning the [InfraGAN repo](https://github.com/makifozkanoglu/InfraGAN),
- Run the below command to apply the patch file changes:
```bash
git apply <relevant changes.patch file>
```

## Setting up the environment
> Conda environment highly preferred over python venv, since some packages are available in conda only. Also for better dependency management.
- After the patch is applied, with this newly available conda `environment.yml` file, run the below command:
```bash
conda env create -f environment.yml
```
- This will create a new conda environment named `thermalgan` with all the required packages.
- There might have issues with CUDA toolkit versions, if latest isn't supported in your device. Check versions and reinstall the relevant pytorch version accordingly.

## Changes made to the original python scripts

### `networks.py`
```python
if isinstance(input, tuple):
    input = input[0]
```
- Added the above line to `networks.py` at line 190 and line 212.
- This was quickfix for the error of tuple being returned instead of tensor. Error shows up when running `train.py`.

### `evaluate.py` (optional)
At line 80, modify to this:
```python
lpips_obj = LPIPS(pretrained=False)
```
Otherwise, you'll need to download pretrained weights of AlexNet.

## Required Dataset Structure

### `vedai` dataset mode
```
dataset_directory/
├── Annotations1024/
│   ├── 00001000.txt
│   ├── 00001001.txt
│   └── ...
├── train/
│   ├── 00001000_co.png
│   ├── 00001000_ir.png
│   ├── 00001001_co.png
│   ├── 00001001_ir.png
│   └── ...
└── test/
    ├── 00001002_co.png
    ├── 00001002_ir.png
    ├── 00001003_co.png
    ├── 00001003_ir.png
    └── ...
```
- Here, `<name>_co.png` is the RGB file, and `<name>_ir.png` is the thermal image file.
- Has been tested to work.

### `kaist` dataset mode
> Note: `kaist` dataset mode is not tested.
```
dataset_directory/
├── annotations/
│   ├── set00/
│   │   ├── V000/
│   │   │   ├── I00000.txt
│   │   │   └── ...
│   └── ...
├── images/
│   ├── set00/
│   │   ├── V000/
│   │   │   ├── lwir/
│   │   │   │   ├── I00000.jpg
│   │   │   │   └── ...
│   │   │   ├── visible/
│   │   │   │   ├── I00000.jpg
│   │   │   │   └── ...
│   └── ...
├── scripts/
│   └── imageSets/
│       ├── train-all-04.txt
│       └── test-all-20.txt
```

## `flir` dataset mode
> Note: Not sure about this either. Inferred from the dataloader scripts.
```
root_directory/
├── grayscale_training_data.npy
├── thermal_training_data.npy
├── grayscale_test_data.npy
└── thermal_test_data.npy
```

## `.vscode/` configuration files
- I have set up `launch.json`, `task.json` files for convenient use of VS Code python debugger, and automating running the visdom server.

## Attempts at running the python scripts
> These are the ones which I was able to get running
### Train Infragan on VEDAI dataset:
```bash
python train.py --dataset_mode VEDAI --dataroot ./datasets/VEDAI --name infragan_vedai --model infragan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4 --save_epoch_freq 1
```

### Train ThermalGAN model on VEDAI dataset:
```bash
python train.py --dataset_mode VEDAI --dataroot ./datasets/fr --name thermal_gan_vedai --model thermal_gan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4 --save_epoch_freq 1
```

### Evaluate ThermalGAN model on VEDAI dataset:
```bash
python evaluate.py --dataset_mode VEDAI --dataroot ./datasets/fr --name thermal_gan_vedai --model thermal_gan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4 --continue_train False --which_epoch latest --results_dir ./results/thermal_gan_vedai/ 
```
- Further details and unsuccessful attempts are mentioned in `thermalgan input options.md`.