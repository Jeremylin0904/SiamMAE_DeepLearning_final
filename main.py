from MaeViT import mae_vit_base_patch16_dec512d8b
from dataset import SiameseDataset
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim

if __name__ == "__main__":

    siamese_dataset = SiameseDataset()
    siamese_dataset.init_items()
    train_dataset = np.array(siamese_dataset.training_vector).transpose(0, 1, 4, 2, 3)
    #print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # Model, optimizer setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mae_vit_base_patch16_dec512d8b().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_idx, (data) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            loss, _, _ = model(data)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

    print("Training complete!")
