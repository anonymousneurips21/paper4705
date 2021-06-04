# Certified Patch Robustness via Smoothed Vision Transformers

This repository contains the code of our **anonymous** NeurIPS submission:

**Certified Patch Robustness via Smoothed Vision Transformers** 

## Getting started
The following steps will get you set up with the required packages:

1.  Clone our repo: `git clone https://github.com/anonymousneurips21/paper4705`

2.  Install dependencies:
    ```
    conda create -n smoothvit python=3.7
    conda activate smoothvit
    pip install robustness
    ```

We will walk you through the steps to create a smoothed ViT on the CIFAR-10 dataset. Similar steps can be followed for other datasets.

### Training the base classifier

The first step is to train the base (ViT) classifier with ablations.
  ```
  python -m src.main.py \
        --dataset cifar10 \ 
        --data /tmp \
        --arch deit_tiny_patch16_224 \
        --pytorch-pretrained \
        --out-dir OUTDIR \
        --exp-name demo \
        --epochs 30 \
        --lr 0.01 \
        --step-lr 10 \
        --batch-size 128 \
        --weight-decay 5e-4 \
        --adv-train 0 \
        --freeze-level -1 \
        --missingness \
        --cifar-preprocess-type simple224 \
        --ablate-input \
        --ablation-type col \
        --ablation-size 4
  ```
Once training is done, the mode is saved in `OUTDIR/demo/`.

### Certifying the smoothed classifier

Now we are ready to apply derandomized smoothing to obtain certificates for each datapoint against adversarial patches. To run certification: 
  ```
  python -m src.main.py \
        --dataset cifar10 \ 
        --data /tmp \
        --arch deit_tiny_patch16_224 \
        --out-dir OUTDIR \
        --exp-name demo \
        --batch-size 128 \
        --adv-train 0 \
        --freeze-level -1 \
        --missingness \
        --cifar-preprocess-type simple224 \
        --resume \
        --certify \
        --certify-out-dir OUTDIR_CERT \
        --certify-mode col \
        --certify-ablation-size 4 \
        --certify-patch-size 5
  ```    

That's it! Now you can replicate all the results of our paper.
