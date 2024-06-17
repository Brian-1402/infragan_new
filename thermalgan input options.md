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


### Test ThermalGAN model on VEDAI dataset (not working)
```bash
python test.py --dataset_mode VEDAI --dataroot ./datasets/VEDAI --name thermal_gan_vedai --model thermal_gan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100 --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4 --continue_train False --which_epoch latest --results_dir ./results/thermal_gan_vedai/ --phase test
```