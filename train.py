import os 
import torch
import torch.optim as optim
import math

def train(model, train_loader, folder_logs, folder_model, num_epochs=20, lr=1e-4, betas=(0.9,0.95), wd=0.05, warmup_epoch=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # If we want to follow the learning schedule of the paper
    # warmup_epoch = 20
    # warmup epochs + cosine decay
    # lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / num_epochs * math.pi) + 1))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    for epoch in range(num_epochs):
        for batch_idx, (data) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            loss, _, _ = model(data)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
        path = os.path.join(folder_model, f'epoch_{epoch}.pt')
        # lr_scheduler.step()
        torch.save(model.state_dict(), path)
        with open(folder_logs, 'a+') as f:
            f.writelines(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()} \n")
    print("Training complete!")
