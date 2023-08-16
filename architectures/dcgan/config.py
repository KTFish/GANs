import torch

# Training
EPOCHS = 100
BATCH_SIZE = 32  # In paper a batch of 128 was used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
LEAKY_RELU_SLOPE = 0.2
HIDDEN_UNITS = 10
Z_DIM = 100


# Optimizer (for details see papers section IV "DETAILS OF ADVERSARIAL TRAINING")
# Oprimizer = ADAM
BETAS = (0.5, 0.999)
LEARNING_RATE = 0.0002
