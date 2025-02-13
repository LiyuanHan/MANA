# Read me 

We used AI data to simulate potential variations in biological data. Specifically, we applied transformations such as translation, rotation, and their combination to MNIST digits, as well as frame-based segmentation for DVS data.

Here, we provide a pre-constructed dataset stored in the `AI_data` directory. However, you are also free to choose your own transformation scales for simulation.

The specific steps are as follows:

1) Run `data_transform.py` to generate `.npz` files with different transformation scales.

2) Run `data_concat.py` to create `.pt` files, which will be used as input to the network.

