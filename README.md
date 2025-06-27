# transformer_scratch
try to implement a vision transformer model from scratch

### dataset
- MNIST (e.g. ylecun/mnist on huggingface or [LINK](https://huggingface.co/datasets/ylecun/mnist))

### performance
- after several epochs of training on training datasets
- 99.0% on test datasets
- 97.6% on train datasets (make this harder to train than test dataset)
- still not converge, but I stopped here.

### trained model weight here on huggingface
- hiki-t/tf_model_mnist or [LINK](https://huggingface.co/hiki-t/tf_model_mnist/tree/main)

### trained results are on wandb
- https://wandb.ai/htsujimu-ucl/vit_mnist_from_scratch?nw=nwuserhtsujimu

little bit messy with loading and saving files with tm02_toy_model_run2.py