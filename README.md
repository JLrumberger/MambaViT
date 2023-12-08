# MambaViT
ViT architecture with Mamba instead of transformer backbone. The ViT code is based on https://github.com/lucidrains/vit-pytorch. This model can be used like this:

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
