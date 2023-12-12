CUDA_VISIBLE_DEVICES=0,1 python -u train_augpolicy_mixed_noisenet_epsilon.py --batch-size 14 --gpu 0,1 \
                        --data-dir ./datasets/LIP --noisenet-prob 0.25 --log-dir 'LIP_augmix_noise_epsilon' \
