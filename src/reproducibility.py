
"""
Section 5.3: Reproducibility
"""

import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_train_loader(data, batch_size=256, seed=SEED):
    X = torch.tensor(data["X_train"], dtype=torch.float32)
    y = torch.tensor(data["y_train"], dtype=torch.float32)

    dataset = TensorDataset(X, y)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )


def run_short_training(data):
    set_seed(SEED)

    X = torch.tensor(data["X_train"][:2048], dtype=torch.float32)
    y = torch.tensor(data["y_train"][:2048], dtype=torch.float32).reshape(-1, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    losses = []

    for epoch in range(3):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


if __name__ == "__main__":
    from src.dl_data import load_dl_data

    set_seed(SEED)
    data = load_dl_data()

    # Test 1: DataLoader reproducibility
    loader1 = make_train_loader(data)
    loader2 = make_train_loader(data)

    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    assert torch.equal(batch1[0], batch2[0])
    assert torch.equal(batch1[1], batch2[1])

    print("Seed:", SEED)
    print("First batch shape:", batch1[0].shape)
    print("Loader reproducibility check PASSED.")

    # Test 2: Training reproducibility
    training_loss_1 = run_short_training(data)
    training_loss_2 = run_short_training(data)

    print("\nTraining run 1:", training_loss_1)
    print("Training run 2:", training_loss_2)

    assert np.allclose(
        training_loss_1, training_loss_2,
        rtol=1e-5, atol=1e-7
    )

    print("Training reproducibility check PASSED.")
