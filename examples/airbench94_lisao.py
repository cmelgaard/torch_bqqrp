"""
airbench94_lisao.py

PLEASE SEE: https://github.com/KellerJordan/cifar10-airbench/blob/28bff5f5b31e95aa45b5b20e1f48baf1ed98d5f6/airbench94_muon.py#L362
The idea of this file is to benchmark out optimizer against Muon on the CIFAR-10 data like above.

Compare SGD vs SingleDeviceLisao (LISAO: momentum regularized by QR/QRCP via
RandLAPACK BQRRP) on CIFAR-10, in an "airbench-style" setup.

- Builds a CIFAR-10 CNN
- Creates two optimizers:
    1) SGD with momentum
    2) SingleDeviceLisao
- Runs a short training loop for each optimizer
- Measures runtime with CUDA events (if CUDA available)
- Prints final test accuracy and total time for each optimizer
"""

#############################################
#                  Setup                    #
#############################################

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from lisao import SingleDeviceLisao


torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


#############################################
#                 Model                     #
#############################################

class SimpleCIFARNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def reset(self):
        """
        Reinitialize all weights/biases (airbench-style model.reset()).
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


#############################################
#               DataLoader                  #
#############################################

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def make_dataloaders(data_root: str, batch_size: int = 256):
    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tfms
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tfms
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, test_loader


#############################################
#              Train / Eval                 #
#############################################

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen


def run_with_optimizer(model, train_loader, test_loader, optimizer, device, name: str):
    """
    Run a short training session with the given optimizer,
    timing GPU code with CUDA events when available.
    """
    # For timing
    use_cuda_timing = (device == "cuda")
    time_seconds = 0.0

    if use_cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        def start_timer():
            starter.record()

        def stop_timer():
            nonlocal time_seconds
            ender.record()
            torch.cuda.synchronize()
            time_seconds += 1e-3 * starter.elapsed_time(ender)
    else:
        t0 = None

        def start_timer():
            nonlocal t0
            t0 = time.perf_counter()

        def stop_timer():
            nonlocal time_seconds, t0
            t1 = time.perf_counter()
            time_seconds += (t1 - t0)

    # Store initial learning rates on param groups (airbench-style)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # Simple schedule: linear decay over a fixed number of steps
    total_train_steps = 5 * len(train_loader)  # short run for comparison

    model.reset()
    model.to(device)

    step = 0
    best_acc = 0.0

    for epoch in range(math.ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            loss.backward()

            # Simple global linear LR decay based on initial_lr
            for group in optimizer.param_groups:
                t = min(1.0, step / max(1, total_train_steps))
                group["lr"] = group["initial_lr"] * (1.0 - t)

            optimizer.step()
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        val_loss, val_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, val_acc)

        print(
            f"[{name}] Epoch {epoch:02d} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3%}, "
            f"best_acc={best_acc:.3%}"
        )

        if step >= total_train_steps:
            break

    print(f"[{name}] Total time: {time_seconds:.3f} seconds")
    return best_acc, time_seconds


#############################################
#                  Main                     #
#############################################

def main():
    device = get_device()
    print(f"Using device: {device}")

    data_root = os.environ.get("CIFAR_DATA", "./data")
    train_loader, test_loader = make_dataloaders(data_root, batch_size=512)

    # Base model definition
    base_model = SimpleCIFARNet(num_classes=10).to(device)
    base_model.reset()

    # Parameter configs (analogous to airbench94_muon.py idea, but simpler)
    params = [p for p in base_model.parameters() if p.requires_grad]

    # Optimizer 1: SGD with momentum
    optimizer1 = torch.optim.SGD(
        params,
        lr=0.1,
        momentum=0.85,
        nesterov=True,
    )

    # Optimizer 2: SingleDeviceLisao (LISAO)
    optimizer2 = SingleDeviceLisao(
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
    )

    optimizers = [optimizer1, optimizer2]
    names = ["SGD", "SingleDeviceLisao"]

    # Compare optimizers: run a separate training pass for each
    results = []
    for opt, name in zip(optimizers, names):
        # Fresh model each time to make the comparison fair
        model = SimpleCIFARNet(num_classes=10).to(device)
        model.reset()
        best_acc, total_time = run_with_optimizer(
            model, train_loader, test_loader, opt, device, name
        )
        results.append((name, best_acc, total_time))

    print("\n==== Summary ====")
    for name, acc, t in results:
        print(f"{name:20s} | best_acc={acc:.3%} | time={t:.3f} s")


if __name__ == "__main__":
    main()
