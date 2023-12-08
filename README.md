# MambaViT
ViT architecture with Mamba instead of transformer backbone. The ViT code is based on https://github.com/lucidrains/vit-pytorch. I only installed it on a Linux server with an Nvidia A40 GPU and CUDA 12.2 which worked fine for me. A short training script with MNIST can be found in `mamba_vit_MNIST_example.ipynb`, it trains to ~90% validation accuracy in a couple of epochs, which is comparable to small ViTs.

This model can be used like this:

```python
m = MambaViT(
    image_size=28,
    patch_size=4,
    num_classes=10,
    channels=1,
    n_layer=8,
    dim=32,
    pool="mean" # mean or cls
    )

img = torch.rand(16,1,28,28)
pred = m(img) # 16 x num_classes
```

