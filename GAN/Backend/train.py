import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------------------------------------
# DEVICE
# -------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------------------------------------
# DATASET: CIFAR-10 (animals + automobile + truck)
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

# -------------------------------------------------------------
# GENERATOR
# -------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.ReLU(True),

            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# -------------------------------------------------------------
# DISCRIMINATOR
# -------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# INITIALIZE MODELS
# -------------------------------------------------------------
latent_dim = 100
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# -------------------------------------------------------------
# SAMPLE FOLDER
# -------------------------------------------------------------
sample_folder = "samples2"
os.makedirs(sample_folder, exist_ok=True)

# -------------------------------------------------------------
# SAVE SAMPLE IMAGES
# -------------------------------------------------------------
def save_samples(epoch, fixed_noise):
    G.eval()
    with torch.no_grad():
        fake = G(fixed_noise).cpu()
        fake = (fake + 1) / 2  

    fig = plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(fake[i].permute(1, 2, 0))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{sample_folder}/epoch_{epoch}.png")
    plt.close()
    G.train()

# -------------------------------------------------------------
# TRAIN LOOP
# -------------------------------------------------------------
epochs = 50
fixed_noise = torch.randn(25, latent_dim, device=device)

print("Starting Training...")

for epoch in range(1, epochs + 1):
    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", colour="magenta")

    for real, _ in progress:
        real = real.to(device)
        batch_size = real.size(0)

        #####################################
        # Train Discriminator
        #####################################

        noise = torch.randn(batch_size, latent_dim, device=device)
        fake = G(noise)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        opt_D.zero_grad()

        real_pred = D(real)
        fake_pred = D(fake.detach())

        loss_real = criterion(real_pred, real_labels)
        loss_fake = criterion(fake_pred, fake_labels)
        loss_D = loss_real + loss_fake

        loss_D.backward()
        opt_D.step()

        #####################################
        # Train Generator
        #####################################

        opt_G.zero_grad()
        fake_pred = D(fake)

        loss_G = criterion(fake_pred, real_labels)
        loss_G.backward()
        opt_G.step()

        progress.set_postfix(G_loss=float(loss_G), D_loss=float(loss_D))

    # Save sample images every 10 epochs
    if epoch % 10 == 0:
        save_samples(epoch, fixed_noise)

    # Optional checkpoint
    torch.save({
        "epoch": epoch,
        "generator_state": G.state_dict(),
        "discriminator_state": D.state_dict(),
        "opt_G_state": opt_G.state_dict(),
        "opt_D_state": opt_D.state_dict()
    }, f"checkpoint_epoch_{epoch}.pth")

# -------------------------------------------------------------
# FINAL MODEL SAVE
# -------------------------------------------------------------
torch.save(G.state_dict(), "generator_final.pth")
torch.save(D.state_dict(), "discriminator_final.pth")

print("Training complete. Models saved.")
