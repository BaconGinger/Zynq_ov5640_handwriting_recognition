import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset  # Import Dataset
import torchvision.transforms as transforms  # Import transforms

# ============================================================
# Dataset paths
# ============================================================

DEFAULT_TRAIN_IMAGES_PATH = "./data/MNIST/raw/train-images.pt"
DEFAULT_TRAIN_LABELS_PATH = "./data/MNIST/raw/train-labels.pt"
DEFAULT_TEST_IMAGES_PATH = "./data/MNIST/raw/t10k-images.pt"
DEFAULT_TEST_LABELS_PATH = "./data/MNIST/raw/t10k-labels.pt"


# ============================================================
# Preprocessing
# ============================================================

def normalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """
    MNIST pixel values:
      0   = background / paper
      255 = digit ink

    HLS expects:
      0.0 = paper
      1.0 = ink
    """
    x = x.float()

    # 如果数据已经是 [0,1]，就不重复除以 255
    if x.max() > 1.0:
        x = x / 255.0

    return x


# 修改后的 preprocess 函数，只处理维度，不执行展平，因为展平会在 Dataset 或 main 中进行
def preprocess_dimensions(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure image tensor is in (N, H, W) format, handling (N, C, H, W) or (N, 784).
    Assumes pixels are already normalized to [0, 1].
    """
    if x.dim() == 4:
        # [N, C, H, W] -> [N, H, W] (assuming C=1 for grayscale)
        x = x.squeeze(1)
    elif x.dim() == 2:
        # [N, 784] -> [N, 28, 28]
        if x.size(1) == 784:
            x = x.view(x.size(0), 28, 28)
        else:
            raise ValueError(f"Unsupported 2D image tensor shape: {tuple(x.shape)}. Expected (N, 784).")
    elif x.dim() == 3:
        # Already [N, H, W], no change needed
        pass
    else:
        raise ValueError(f"Unsupported image tensor shape: {tuple(x.shape)}")
    return x


# ============================================================
# PyTorch Model
# ============================================================

class ReluSigmoidNN(nn.Module):
    """
    784 -> 64 -> 32 -> 24 -> 20 -> 16 -> 10

    Hidden layers:
      ReLU

    Output:
      During training, forward returns logits.
      BCEWithLogitsLoss internally applies sigmoid.

    For inference:
      predict() uses argmax(logits), equivalent to argmax(sigmoid(logits)).
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 20)
        self.fc5 = nn.Linear(20, 16)
        self.fc6 = nn.Linear(16, 10)

        self.reset_parameters()

    def reset_parameters(self):
        """
        He/Kaiming initialization for ReLU hidden layers.
        Output layer uses Xavier initialization.
        Bias initialized to zero.
        """

        hidden_layers = [
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4,
            self.fc5,
        ]

        for layer in hidden_layers:
            nn.init.kaiming_uniform_(
                layer.weight,
                nonlinearity="relu",
            )
            nn.init.zeros_(layer.bias)

        # Output layer is linear logits before sigmoid/BCEWithLogitsLoss.
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        logits = self.fc6(x)
        return logits

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ============================================================
# Custom Dataset for Augmentation
# ============================================================

class MNISTAugmentationDataset(Dataset):
    """
    A custom Dataset to apply transformations to MNIST images (H, W)
    and then flatten them to 784 pixels.
    """

    def __init__(self, images_hw, labels, transform):
        # images_hw: N x H x W tensor (already normalized to [0, 1] float)
        self.images_hw = images_hw
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_hw = self.images_hw[idx]  # (H, W) tensor
        label = self.labels[idx]

        # torchvision transforms expect [C, H, W] for tensors.
        # MNIST is grayscale, so add a channel dimension.
        img_chw = img_hw.unsqueeze(0)  # -> (1, H, W)

        if self.transform:
            img_chw = self.transform(img_chw)

        # Flatten to 784 for the model input
        img_flat = img_chw.view(-1)  # -> (784,)

        return img_flat, label


# ============================================================
# Load MNIST .pt dataset
# ============================================================

def torch_load_compat(path):
    """
    Compatible torch.load for different PyTorch versions.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ensure_tensor(obj, name="data"):
    """
    Convert loaded object to torch.Tensor.

    Some .pt files may directly store Tensor.
    Some may store numpy arrays or list.
    """
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)

    if isinstance(obj, list):
        return torch.tensor(obj)

    raise TypeError(f"Unsupported {name} type: {type(obj)}")


def load_mnist_from_pt(
        train_images_path=DEFAULT_TRAIN_IMAGES_PATH,
        train_labels_path=DEFAULT_TRAIN_LABELS_PATH,
        test_images_path=DEFAULT_TEST_IMAGES_PATH,
        test_labels_path=DEFAULT_TEST_LABELS_PATH,
):
    print("=" * 60)
    print("  Loading MNIST dataset from .pt files")
    print("=" * 60)

    print(f"  Train images: {train_images_path}")
    print(f"  Train labels: {train_labels_path}")
    print(f"  Test images:  {test_images_path}")
    print(f"  Test labels:  {test_labels_path}")

    train_images = ensure_tensor(torch_load_compat(train_images_path), "train_images")
    train_labels = ensure_tensor(torch_load_compat(train_labels_path), "train_labels")
    test_images = ensure_tensor(torch_load_compat(test_images_path), "test_images")
    test_labels = ensure_tensor(torch_load_compat(test_labels_path), "test_labels")

    # Normalize pixel values to [0, 1]
    train_images = normalize_mnist(train_images)
    test_images = normalize_mnist(test_images)

    # Ensure images are (N, H, W) before returning
    train_images = preprocess_dimensions(train_images)
    test_images = preprocess_dimensions(test_images)

    train_labels = train_labels.long().view(-1)
    test_labels = test_labels.long().view(-1)

    print()
    print(f"  Train images shape: {tuple(train_images.shape)} (H, W tensors)")
    print(f"  Train labels shape: {tuple(train_labels.shape)}")
    print(f"  Test images shape:  {tuple(test_images.shape)} (H, W tensors)")
    print(f"  Test labels shape:  {tuple(test_labels.shape)}")

    active = torch.mean((train_images > 0.1).float(), dim=(1, 2)) * 784  # Adjust for H,W dimensions
    print(f"  Pixels with value > 0.1: {active.mean().item():.0f}/784 avg")
    print()

    return train_images, train_labels, test_images, test_labels


# ============================================================
# Accuracy / Evaluation
# ============================================================

@torch.no_grad()
def accuracy(model, x_flat, y, device, batch_size=1024):
    model.eval()

    # For evaluation, num_workers can typically be 0 as there's no expensive augmentation
    # and x_flat is already a ready-to-use tensor.
    # If evaluation is also very slow on large datasets, you might increase this.
    num_workers_eval = 0
    pin_memory_eval = device.type == 'cuda' and num_workers_eval > 0

    dataset = TensorDataset(x_flat, y)  # Expects flattened images for evaluation
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_eval,  # Added for evaluation
        pin_memory=pin_memory_eval,  # Added for evaluation
    )

    total = 0
    correct = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        preds = model.predict(xb)

        total += yb.numel()
        correct += (preds == yb).sum().item()

    return correct / total


@torch.no_grad()
def run_test(model, x_test_flat, y_test, device, batch_size=1024):
    print("=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)

    model.eval()

    test_acc = accuracy(model, x_test_flat, y_test, device, batch_size=batch_size)
    n_correct = int(test_acc * len(y_test))

    print(f"\n  Overall accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Correct / Total:  {n_correct} / {len(y_test)}")

    # For evaluation, num_workers can typically be 0 as there's no expensive augmentation
    num_workers_eval = 0
    pin_memory_eval = device.type == 'cuda' and num_workers_eval > 0

    dataset = TensorDataset(x_test_flat, y_test)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_eval,  # Added for evaluation
        pin_memory=pin_memory_eval,  # Added for evaluation
    )

    all_preds = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        preds = model.predict(xb).cpu()

        all_preds.append(preds)
        all_labels.append(yb)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    print("\n  Per-class accuracy:")
    for c in range(10):
        mask = labels == c
        if mask.sum().item() > 0:
            c_acc = (preds[mask] == c).float().mean().item()
            bar = "#" * int(c_acc * 30) + "." * (30 - int(c_acc * 30))
            print(f"    {c}: {bar} {c_acc * 100:.1f}%")

    print()


# ============================================================
# Training
# ============================================================

def run_training(model, train_loader, x_train_eval_flat, y_train_eval, x_val_eval_flat, y_val_eval, args, device):
    """
    Train PyTorch model.

    Uses:
      BCEWithLogitsLoss
      SGD + Momentum
      Exponential LR decay
    """

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.lr_decay,
    )

    best_val_acc = 0.0
    best_state = None

    print("=" * 60)
    print("  TRAINING")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Momentum:   {args.momentum}")
    print(f"  LR decay:   {args.lr_decay}")
    if args.augment:
        print(f"  Data Augmentation: ENABLED")
    else:
        print(f"  Data Augmentation: DISABLED")
    print(f"  Num Workers: {train_loader.num_workers}")  # Print num_workers
    print(f"  Pin Memory:  {train_loader.pin_memory}")  # Print pin_memory
    print()

    print("  " + "-" * 72)
    print(
        f"  {'Epoch':>5s} | "
        f"{'Train Loss':>10s} | "
        f"{'Train Acc':>9s} | "
        f"{'Val Acc':>8s} | "
        f"{'LR':>10s} | "
        f"{'Time':>7s}"
    )
    print("  " + "-" * 72)

    for epoch in range(args.epochs):
        t0 = time.time()

        model.train()
        total_loss = 0.0
        total_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            y_onehot = torch.zeros(
                yb.size(0),
                10,
                dtype=torch.float32,
                device=device,
            )
            y_onehot.scatter_(1, yb.view(-1, 1), 1.0)

            logits = model(xb)
            loss = criterion(logits, y_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        # Use the original (non-augmented) training and validation sets for evaluation
        train_acc = accuracy(
            model,
            x_train_eval_flat,
            y_train_eval,
            device,
            batch_size=args.eval_batch_size,
        )
        val_acc = accuracy(
            model,
            x_val_eval_flat,
            y_val_eval,
            device,
            batch_size=args.eval_batch_size,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"  {epoch + 1:5d} | "
            f"{avg_loss:10.6f} | "
            f"{train_acc:9.4f} | "
            f"{val_acc:8.4f} | "
            f"{current_lr:10.6f} | "
            f"{elapsed:6.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        scheduler.step()

    print("  " + "-" * 72)
    print(f"\n  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc


# ============================================================
# Weight loading from .dat
# ============================================================

def load_matrix(fpath, shape):
    flat = []

    with open(fpath, "r") as f:
        text = f.read()

    parts = text.replace("\r", "").replace("\n", "").split(",")

    for p in parts:
        p = p.strip()
        if p:
            flat.append(float(p))

    arr = np.array(flat, dtype=np.float32).reshape(shape)
    return arr


def load_vector(fpath):
    flat = []

    with open(fpath, "r") as f:
        text = f.read()

    parts = text.replace("\r", "").replace("\n", "").split(",")

    for p in parts:
        p = p.strip()
        if p:
            flat.append(float(p))

    return np.array(flat, dtype=np.float32)


def load_weights(model, src_dir, device):
    """
    Load HLS-compatible .dat files into PyTorch model.

    HLS matrix shape:
      weight1: [784, 64]

    PyTorch Linear weight shape:
      fc1.weight: [64, 784]

    Therefore:
      PyTorch weight = HLS weight.T
    """
    print(f"  Loading weights from: {src_dir}")

    w1 = load_matrix(os.path.join(src_dir, "weight1.dat"), (784, 64))
    w2 = load_matrix(os.path.join(src_dir, "weight2.dat"), (64, 32))
    w3 = load_matrix(os.path.join(src_dir, "weight3.dat"), (32, 24))
    w4 = load_matrix(os.path.join(src_dir, "weight4.dat"), (24, 20))
    w5 = load_matrix(os.path.join(src_dir, "weight5.dat"), (20, 16))
    w6 = load_matrix(os.path.join(src_dir, "weight6.dat"), (16, 10))

    b1 = load_vector(os.path.join(src_dir, "bias1.dat"))
    b2 = load_vector(os.path.join(src_dir, "bias2.dat"))
    b3 = load_vector(os.path.join(src_dir, "bias3.dat"))
    b4 = load_vector(os.path.join(src_dir, "bias4.dat"))
    b5 = load_vector(os.path.join(src_dir, "bias5.dat"))
    b6 = load_vector(os.path.join(src_dir, "bias6.dat"))

    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(w1.T))
        model.fc2.weight.copy_(torch.from_numpy(w2.T))
        model.fc3.weight.copy_(torch.from_numpy(w3.T))
        model.fc4.weight.copy_(torch.from_numpy(w4.T))
        model.fc5.weight.copy_(torch.from_numpy(w5.T))
        model.fc6.weight.copy_(torch.from_numpy(w6.T))

        model.fc1.bias.copy_(torch.from_numpy(b1))
        model.fc2.bias.copy_(torch.from_numpy(b2))
        model.fc3.bias.copy_(torch.from_numpy(b3))
        model.fc4.bias.copy_(torch.from_numpy(b4))
        model.fc5.bias.copy_(torch.from_numpy(b5))
        model.fc6.bias.copy_(torch.from_numpy(b6))

    model.to(device)

    print("  Weights loaded successfully.")


# ============================================================
# Weight export to .dat
# ============================================================

def export_matrix(fpath, matrix):
    """
    Export a 2D matrix in row-major order.

    C code:
      const float weight1[784][64] = {
          #include "weight1.dat"
      };

    PyTorch Linear weight is [out_features, in_features].
    HLS expects [in_features, out_features].

    So before calling this function, transpose PyTorch weights.
    """
    flat = matrix.flatten()

    with open(fpath, "w", newline="") as f:
        for i, val in enumerate(flat):
            f.write(f"{val:.6f}")
            if i < len(flat) - 1:
                f.write(",\r\n")

    size_kb = os.path.getsize(fpath) / 1024
    print(
        f"  {os.path.basename(fpath):20s} "
        f"{matrix.shape[0]:5d}×{matrix.shape[1]:<5d} "
        f"{size_kb:8.1f} KB"
    )


def export_vector(fpath, vector):
    flat = vector.flatten()

    with open(fpath, "w", newline="") as f:
        for i, val in enumerate(flat):
            f.write(f"{val:.6f}")
            if i < len(flat) - 1:
                f.write(",\r\n")

    size_b = os.path.getsize(fpath)
    print(
        f"  {os.path.basename(fpath):20s} "
        f"{len(flat):5d} elements  "
        f"{size_b:6d} B"
    )


def export_weights(model, out_dir):
    """
    Export PyTorch model weights to HLS-compatible .dat files.

    HLS expects:
      W1 shape: [784, 64]
      W2 shape: [64, 32]
      W3 shape: [32, 24]
      W4 shape: [24, 20]
      W5 shape: [20, 16]
      W6 shape: [16, 10]

    PyTorch Linear uses:
      weight shape: [out_features, in_features]

    Therefore export:
      fc.weight.T
    """
    print("\nExporting weights/biases...")
    os.makedirs(out_dir, exist_ok=True)

    model_cpu = model.cpu()
    model_cpu.eval()

    with torch.no_grad():
        w1 = model_cpu.fc1.weight.detach().numpy().T
        w2 = model_cpu.fc2.weight.detach().numpy().T
        w3 = model_cpu.fc3.weight.detach().numpy().T
        w4 = model_cpu.fc4.weight.detach().numpy().T
        w5 = model_cpu.fc5.weight.detach().numpy().T
        w6 = model_cpu.fc6.weight.detach().numpy().T

        b1 = model_cpu.fc1.bias.detach().numpy()
        b2 = model_cpu.fc2.bias.detach().numpy()
        b3 = model_cpu.fc3.bias.detach().numpy()
        b4 = model_cpu.fc4.bias.detach().numpy()
        b5 = model_cpu.fc5.bias.detach().numpy()
        b6 = model_cpu.fc6.bias.detach().numpy()

    export_matrix(os.path.join(out_dir, "weight1.dat"), w1)
    export_matrix(os.path.join(out_dir, "weight2.dat"), w2)
    export_matrix(os.path.join(out_dir, "weight3.dat"), w3)
    export_matrix(os.path.join(out_dir, "weight4.dat"), w4)
    export_matrix(os.path.join(out_dir, "weight5.dat"), w5)
    export_matrix(os.path.join(out_dir, "weight6.dat"), w6)

    export_vector(os.path.join(out_dir, "bias1.dat"), b1)
    export_vector(os.path.join(out_dir, "bias2.dat"), b2)
    export_vector(os.path.join(out_dir, "bias3.dat"), b3)
    export_vector(os.path.join(out_dir, "bias4.dat"), b4)
    export_vector(os.path.join(out_dir, "bias5.dat"), b5)
    export_vector(os.path.join(out_dir, "bias6.dat"), b6)

    print("  Done.")


# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch train script for 5tanhHuanip HLS project"
    )

    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_IMAGES_PATH)
    parser.add_argument("--train_labels_path", type=str, default=DEFAULT_TRAIN_LABELS_PATH)
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_IMAGES_PATH)
    parser.add_argument("--test_labels_path", type=str, default=DEFAULT_TEST_LABELS_PATH)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    # Data Augmentation Arguments
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation for training data.")
    parser.add_argument("--augment_rot_degrees", type=float, nargs=2, default=(-10, 10),
                        help="Degrees for random rotation (min_deg, max_deg). Default: (-10, 10)")
    parser.add_argument("--augment_translate_percent", type=float, nargs=2, default=(0.05, 0.05),
                        help="Max percentage for random translation (max_dx_ratio, max_dy_ratio). Default: (0.05, 0.05)")
    parser.add_argument("--augment_scale_range", type=float, nargs=2, default=(0.9, 1.1),
                        help="Scale factor range for random scaling (min_scale, max_scale). Default: (0.9, 1.1)")
    parser.add_argument("--augment_shear_degrees", type=float, nargs=2, default=(-5, 5),
                        help="Degrees for random shear (min_shear_deg, max_shear_deg). Default: (-5, 5)")
    parser.add_argument("--augment_blur_sigma", type=float, nargs=2, default=(0.1, 1.0),
                        help="Sigma range for random Gaussian blur (min_sigma, max_sigma). Default: (0.1, 1.0)")

    # New argument for controlling DataLoader num_workers
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of data loading workers. Default: (CPU count - 1) for CUDA, 0 for CPU.")

    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hls_1",
            "src",
            "weights_export",
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  PyTorch NN: 784->64->32->24->20->16->10, hidden=ReLU, output=Sigmoid")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Seed:   {args.seed}")
    print()

    # Load images as [N, H, W] tensors (normalized [0,1])
    x_train_hw, y_train, x_test_hw, y_test = load_mnist_from_pt(
        train_images_path=args.train_path,
        train_labels_path=args.train_labels_path,
        test_images_path=args.test_path,
        test_labels_path=args.test_labels_path,
    )

    # Always flatten original training and test images for evaluation (no augmentation here)
    x_train_eval_flat = x_train_hw.view(x_train_hw.size(0), -1)
    x_test_eval_flat = x_test_hw.view(x_test_hw.size(0), -1)

    model = ReluSigmoidNN()

    if args.export_only:
        export_weights(model, args.out_dir)
        print("\n[OK] Initialized weights exported, not trained.")
        return

    if args.test:
        try:
            load_weights(model, args.out_dir, device)
            run_test(
                model,
                x_test_eval_flat,  # Use original test set for testing
                y_test,
                device,
                batch_size=args.eval_batch_size,
            )
        except FileNotFoundError:
            print(f"  *** ERROR: Weight files not found in {args.out_dir} ***")
            print("  *** Please run training first. ***")
        except Exception as e:
            print(f"  *** ERROR loading/testing weights: {e} ***")
        return

    # Determine num_workers and pin_memory based on device and user input
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        if device.type == 'cuda' and os.cpu_count() > 1:
            num_workers = os.cpu_count() - 1  # Use all but one CPU core for CUDA
            print(f"  Automatically setting num_workers to {num_workers} for CUDA training (CPU count - 1).")
        else:
            num_workers = 0  # Default to 0 for CPU training or if only 1 CPU core
            print(f"  Setting num_workers to {num_workers} (default for CPU or single-core).")

    pin_memory = device.type == 'cuda' and num_workers > 0
    if pin_memory:
        print("  Pin memory is ENABLED for CUDA training.")
    else:
        print("  Pin memory is DISABLED.")

    # Prepare training dataset and loader, with or without augmentation
    if args.augment:
        print("  Applying data augmentation to training data.")
        transform_list = []
        if args.augment_rot_degrees:
            transform_list.append(transforms.RandomAffine(
                degrees=args.augment_rot_degrees,
                interpolation=transforms.InterpolationMode.BILINEAR))
        if args.augment_translate_percent:
            # RandomAffine takes translate as (max_dx, max_dy) relative to image size
            transform_list.append(transforms.RandomAffine(
                translate=args.augment_translate_percent,
                interpolation=transforms.InterpolationMode.BILINEAR, degrees=0))
        if args.augment_scale_range:
            transform_list.append(transforms.RandomAffine(
                scale=args.augment_scale_range,
                interpolation=transforms.InterpolationMode.BILINEAR, degrees=0))
        if args.augment_shear_degrees:
            transform_list.append(transforms.RandomAffine(
                shear=args.augment_shear_degrees,
                interpolation=transforms.InterpolationMode.BILINEAR, degrees=0))
        if args.augment_blur_sigma:
            # kernel_size for GaussianBlur must be odd. Using 3x3 for MNIST size.
            transform_list.append(transforms.GaussianBlur(
                kernel_size=3, sigma=args.augment_blur_sigma))

        if not transform_list:  # If augment was true but no specific transforms requested, still flat
            train_dataset = TensorDataset(x_train_eval_flat, y_train)
        else:
            augmentation_transform = transforms.Compose(transform_list)
            train_dataset = MNISTAugmentationDataset(x_train_hw, y_train, augmentation_transform)
    else:
        # If no augmentation, just flatten the original training images for the dataset
        train_dataset = TensorDataset(x_train_eval_flat, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,  # Pass calculated num_workers
        pin_memory=pin_memory,  # Pass calculated pin_memory
    )

    best_val_acc = run_training(
        model,
        train_loader,
        x_train_eval_flat,  # Use original, flattened train set for evaluation
        y_train,
        x_test_eval_flat,  # Use original, flattened test set for evaluation
        y_test,
        args,
        device,
    )

    train_acc = accuracy(
        model,
        x_train_eval_flat,  # Use original, flattened train set for evaluation
        y_train,
        device,
        batch_size=args.eval_batch_size,
    )
    val_acc = accuracy(
        model,
        x_test_eval_flat,  # Use original, flattened test set for evaluation
        y_test,
        device,
        batch_size=args.eval_batch_size,
    )

    print()
    print("Final model:")
    print(f"  Training accuracy:   {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"  Validation accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

    run_test(
        model,
        x_test_eval_flat,  # Use original, flattened test set for evaluation
        y_test,
        device,
        batch_size=args.eval_batch_size,
    )

    export_weights(model, args.out_dir)

    print()
    print(f"[OK] Training complete.")
    print(f"     Best val_acc = {best_val_acc * 100:.2f}%")
    print(f"     Weights saved to: {args.out_dir}")


if __name__ == "__main__":
    main()