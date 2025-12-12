# UNet Binary Segmentation on Oxford-IIIT Pet

This project implements and trains several UNet variants for binary semantic segmentation on the Oxford-IIIT Pet dataset. The task is to segment the pet (including boundary) from the background in 128×128 RGB images.

## Project Overview

The script defines:  
- A custom dataset wrapper around `torchvision.datasets.OxfordIIITPet` that produces binary masks.  
- Modular contracting (encoder) and expanding (decoder) blocks to build UNet variants with different downsampling and upsampling strategies.  
- Two loss functions (binary cross entropy and Dice loss) and a Dice score metric for evaluation.  
- A training loop with train/validation splits and four different UNet configurations trained and saved to disk.

## Dataset and Preprocessing

The dataset is the Oxford-IIIT Pet dataset loaded from `torchvision.datasets.OxfordIIITPet` with segmentation masks.

- **Custom wrapper**: `OxfordPetsSegmentation(OxfordIIITPet)`  
  - Arguments: `root='./data'`, `split='trainval'`, `size=128`, `download=True` (default values).  
  - Image transform: resize to 128×128 and convert to tensor.  
  - Mask transform: resize to 128×128 and convert to tensor (kept as single-channel integer labels before binarization).

- **Binary mask creation**:  
  - Original masks contain trimap labels; the code converts them to binary by treating label values `<= 2` as foreground (pet + boundary) and others as background.  
  - Output mask is a float tensor with values 0 or 1 and shape `[1, H, W]`.

- **Data split**:  
  - A `trainval` split is loaded, then split into train and validation with a 90/10 ratio using `random_split`.  
  - Train and validation DataLoaders use batch size `8` and `num_workers=2`.

## Model Architecture

The UNet implementation is modular: contracting blocks for downsampling and expanding blocks for upsampling.

### Contracting (Encoder) Blocks

Two encoder variants are defined, each returning a downsampled tensor and a skip connection feature map.

- **`ContractingBlockMP`** (MaxPool version):  
  - Two convolution–BatchNorm–ReLU layers, followed by a `MaxPool2d` with stride 2.  
  - Signature: `ContractingBlockMP(in_ch, out_ch)`; `forward(x)` returns `(x_down, skip)` where `skip` is the feature before pooling.

- **`ContractingBlockStrided`** (Strided Conv version):  
  - Two convolution–BatchNorm–ReLU layers, followed by a strided convolution (`stride=2`) for downsampling.  
  - Signature: `ContractingBlockStrided(in_ch, out_ch)`; `forward(x)` returns `(x_down, skip)` similar to max-pool variant.

### Expanding (Decoder) Blocks

Two decoder variants are defined, each taking the current feature map and a skip tensor from the encoder.

- **`ExpandingBlockTranspose`** (Transposed Conv):  
  - Upsampling via `ConvTranspose2d` with kernel size 2 and stride 2.  
  - Concatenates the upsampled feature with the skip connection along channel dimension.  
  - Two convolution–BatchNorm–ReLU layers after concatenation.

- **`ExpandingBlockUpsample`** (Nearest-Neighbor + Conv):  
  - Upsampling via `nn.Upsample(scale_factor=2, mode="nearest")`.  
  - Concatenation with the skip, followed by two convolution–BatchNorm–ReLU layers.

### UNet Class

The `UNet` class assembles these blocks into a 3-level encoder–decoder network.

- **Signature**: `UNet(contracting, expanding, in_channels=3, out_channels=1)` where `contracting` and `expanding` are class references to one of the encoder/decoder variants.  
- **Encoder**:  
  - `contract1`: `in_channels → 64`  
  - `contract2`: `64 → 128`  
  - `contract3`: `128 → 256`

- **Decoder**:  
  - `expand1`: upsample 256, concatenate with 256-channel skip, output 256 channels.  
  - `expand2`: upsample 256, concatenate with 128-channel skip, output 128 channels.  
  - `expand3`: upsample 128, concatenate with 64-channel skip, output 64 channels.

- **Final layer**: `Conv2d(64, out_channels, kernel_size=1)` followed by `sigmoid` for binary segmentation.

**Forward pass**:  
1. Pass input through three contracting blocks to obtain three skip tensors and a bottleneck feature.  
2. Decode via three expanding blocks using corresponding skip connections.  
3. Apply `final_conv` and sigmoid to produce `[B, 1, 128, 128]` predictions in `[0,1]`.

## Loss Functions and Metrics

Two loss functions and a Dice metric are defined.

- **Binary Cross Entropy loss**:  
  - `bce_loss(pred, target)` wraps `nn.BCELoss()`.  
  - Used when treating output probabilities directly against 0/1 masks.

- **Dice loss**:  
  - `dice_loss(pred, target)` flattens both tensors, computes intersection and union, and returns `1 - dice_coefficient`.  
  - Includes a small `smooth = 1e-6` to avoid division by zero.

- **Dice score metric** (for evaluation):  
  - `dice_score(pred, target)` binarizes predictions at 0.5 threshold, computes per-batch Dice, and returns mean Dice as a scalar Python float.

## Training Loop

The core training loop is encapsulated in `train_model`.

### Function: `train_model`

**Signature**:  

train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs)


**Behavior**:  
- Tracks `best_val_dice` over epochs.  
- **Training phase per epoch**:  
  - Sets `model.train()`.  
  - For each batch: moves data to device, zeroes gradients, runs forward pass, computes loss, backpropagates, and steps the optimizer.  
  - Uses `tqdm` progress bar with current train loss.  
- **Validation phase per epoch**:  
  - Sets `model.eval()` and disables gradients.  
  - Computes validation loss and accumulates Dice score over the validation loader.  
  - Updates `best_val_dice` if current epoch's validation Dice improves.  
- Prints epoch-level training loss, validation loss, and validation Dice.  
- Returns `best_val_dice` for the model at the end of training.

## Main Script Behavior

The bottom of the file contains the `if __name__ == "__main__":` section that prepares data, runs experiments, and saves models.

### Environment and Paths

- Attempts to mount Google Drive (Colab environment) under `/content/drive`, then sets `drive_root` to `'/content/drive/MyDrive/ERAV4/Session_15'`.  
- Creates and changes to that directory, and sets `data_path` accordingly when the mount is successful.  
- If Drive mount fails (e.g., running locally), falls back to `data_path = './data'`.  
- Detects device as `"cuda"` if available, otherwise `"cpu"`.

### Dataset Loading and Split

- Loads `OxfordPetsSegmentation` with `split='trainval'`, `size=128`, and `download=True`.  
- Prints dataset size and sample shape info.  
- Splits dataset into train and validation subsets with a 90/10 ratio.  
- Constructs `train_loader` and `val_loader` with `batch_size=8`.

### Model Configurations

Four model configurations are defined, corresponding to combinations of:

- Contracting block: max pooling vs strided convolution.  
- Expanding block: transposed convolution vs upsample+conv.  
- Loss function: BCE vs Dice.

**Configurations list**:

| Name              | Contracting block       | Expanding block           | Loss      |
|-------------------|-------------------------|---------------------------|-----------|
| `MP+Tr+BCE`       | `ContractingBlockMP`    | `ExpandingBlockTranspose` | BCE       |
| `MP+Tr+Dice`      | `ContractingBlockMP`    | `ExpandingBlockTranspose` | Dice      |
| `StrConv+Tr+BCE`  | `ContractingBlockStrided` | `ExpandingBlockTranspose` | BCE    |
| `StrConv+Ups+Dice`| `ContractingBlockStrided` | `ExpandingBlockUpsample` | Dice   |

Each configuration name encodes its downsampling, upsampling, and loss choices.

### Training All Variants

For each configuration:

- Instantiate a `UNet` with the specified contracting and expanding block classes.  
- Move the model to the selected device.  
- Use `optim.Adam` with `lr=1e-4`.  
- Call `train_model` with `epochs=10` and the appropriate loss function and data loaders.  
- Save model weights to a `.pth` file named after the configuration (with `+` and spaces removed or replaced).  
- Append the configuration name and best validation Dice to a results list.

After training all four models, the script prints a summary of each configuration's best validation Dice.

### Training Logs

Using device: cuda

Loading Oxford-IIIT Pet Dataset...
(First run will download ~800MB - this may take a few minutes)
✓ Dataset loaded: 3680 images

✓ Dataset processed successfully!
  - Train samples: 3312
  - Val samples:   368
  - First image shape: torch.Size([3, 128, 128])
  - First mask shape:  torch.Size([1, 128, 128])


============================================================
Training Model 1/4: MP+Tr+BCE
============================================================
Epoch 1/10: TrainLoss=0.3377 | ValLoss=0.2886 | ValDice=0.9423
Epoch 2/10: TrainLoss=0.2525 | ValLoss=0.2456 | ValDice=0.9460
Epoch 3/10: TrainLoss=0.2236 | ValLoss=0.2251 | ValDice=0.9504
Epoch 4/10: TrainLoss=0.2083 | ValLoss=0.2194 | ValDice=0.9511
Epoch 5/10: TrainLoss=0.1971 | ValLoss=0.2074 | ValDice=0.9525
Epoch 6/10: TrainLoss=0.1897 | ValLoss=0.2022 | ValDice=0.9541
Epoch 7/10: TrainLoss=0.1821 | ValLoss=0.2044 | ValDice=0.9535
Epoch 8/10: TrainLoss=0.1760 | ValLoss=0.1998 | ValDice=0.9541
Epoch 9/10: TrainLoss=0.1691 | ValLoss=0.1978 | ValDice=0.9553
Epoch 10/10: TrainLoss=0.1626 | ValLoss=0.2091 | ValDice=0.9543
  ✓ Model saved: MP_Tr_BCE.pth

============================================================
Training Model 2/4: MP+Tr+Dice
============================================================
Epoch 1/10: TrainLoss=0.1948 | ValLoss=0.1455 | ValDice=0.9398
Epoch 2/10: TrainLoss=0.1225 | ValLoss=0.1052 | ValDice=0.9426
Epoch 3/10: TrainLoss=0.0903 | ValLoss=0.0807 | ValDice=0.9458
Epoch 4/10: TrainLoss=0.0726 | ValLoss=0.0683 | ValDice=0.9482
Epoch 5/10: TrainLoss=0.0623 | ValLoss=0.0615 | ValDice=0.9491
Epoch 6/10: TrainLoss=0.0562 | ValLoss=0.0562 | ValDice=0.9510
Epoch 7/10: TrainLoss=0.0519 | ValLoss=0.0527 | ValDice=0.9523
Epoch 8/10: TrainLoss=0.0489 | ValLoss=0.0511 | ValDice=0.9524
Epoch 9/10: TrainLoss=0.0466 | ValLoss=0.0503 | ValDice=0.9524
Epoch 10/10: TrainLoss=0.0448 | ValLoss=0.0484 | ValDice=0.9536
  ✓ Model saved: MP_Tr_Dice.pth

============================================================
Training Model 3/4: StrConv+Tr+BCE
============================================================
Epoch 1/10: TrainLoss=0.4257 | ValLoss=0.3428 | ValDice=0.9376
Epoch 2/10: TrainLoss=0.3035 | ValLoss=0.2883 | ValDice=0.9436
Epoch 3/10: TrainLoss=0.2565 | ValLoss=0.2472 | ValDice=0.9471
Epoch 4/10: TrainLoss=0.2303 | ValLoss=0.2300 | ValDice=0.9508
Epoch 5/10: TrainLoss=0.2121 | ValLoss=0.2150 | ValDice=0.9520
Epoch 6/10: TrainLoss=0.2012 | ValLoss=0.2186 | ValDice=0.9519
Epoch 7/10: TrainLoss=0.1928 | ValLoss=0.2057 | ValDice=0.9532
Epoch 8/10: TrainLoss=0.1858 | ValLoss=0.2061 | ValDice=0.9541
Epoch 9/10: TrainLoss=0.1799 | ValLoss=0.2041 | ValDice=0.9541
Epoch 10/10: TrainLoss=0.1743 | ValLoss=0.1948 | ValDice=0.9553
  ✓ Model saved: StrConv_Tr_BCE.pth

============================================================
Training Model 4/4: StrConv+Ups+Dice
============================================================
Epoch 1/10: TrainLoss=0.1513 | ValLoss=0.1221 | ValDice=0.9411
Epoch 2/10: TrainLoss=0.0987 | ValLoss=0.0852 | ValDice=0.9431
Epoch 3/10: TrainLoss=0.0774 | ValLoss=0.0726 | ValDice=0.9445
Epoch 4/10: TrainLoss=0.0654 | ValLoss=0.0644 | ValDice=0.9466
Epoch 5/10: TrainLoss=0.0578 | ValLoss=0.0590 | ValDice=0.9476
Epoch 6/10: TrainLoss=0.0531 | ValLoss=0.0535 | ValDice=0.9505
Epoch 7/10: TrainLoss=0.0499 | ValLoss=0.0513 | ValDice=0.9516
Epoch 8/10: TrainLoss=0.0478 | ValLoss=0.0502 | ValDice=0.9518
Epoch 9/10: TrainLoss=0.0460 | ValLoss=0.0496 | ValDice=0.9519
Epoch 10/10: TrainLoss=0.0443 | ValLoss=0.0469 | ValDice=0.9541
  ✓ Model saved: StrConv_Ups_Dice.pth

============================================================
TRAINING SUMMARY - ALL 4 MODELS COMPLETED
============================================================
MP+Tr+BCE            - Best Val Dice: 0.9553
MP+Tr+Dice           - Best Val Dice: 0.9536
StrConv+Tr+BCE       - Best Val Dice: 0.9553
StrConv+Ups+Dice     - Best Val Dice: 0.9541
============================================================
