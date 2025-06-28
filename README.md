# transformer_scratch
try to implement a vision transformer model from scratch with purely torch packages

### dataset
- MNIST (e.g. ylecun/mnist on huggingface or [LINK](https://huggingface.co/datasets/ylecun/mnist))

### performance

- Step 01 - One digit image task
    - after several epochs of training on training datasets
    - 99.0% on test datasets
    - 97.6% on train datasets (make this harder to train than test dataset)
    - still not converge, but I stopped here.
- Step 02 - A sequence of 4 digits image task
- for decoder model, I used RNN with CTC loss ([LINK](https://distill.pub/2017/ctc/))
    - after 1 epoch of training, reached 94% on test datasets
    - after 3 extra epochs of training, reached 95% on test datasets
    - still not converge, but I stopped here (RNN training takes time!)

### trained model weight here on huggingface
- hiki-t/tf_model_mnist or [LINK](https://huggingface.co/hiki-t/tf_model_mnist/tree/main)

### trained results are on wandb
- https://wandb.ai/htsujimu-ucl/vit_mnist_from_scratch?nw=nwuserhtsujimu

little bit messy with loading and saving files with tm02_toy_model_run2.py