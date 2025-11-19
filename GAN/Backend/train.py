import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# AMP is only allowed on CUDA
use_amp = device == "cuda"
if device != "cuda":
    print("⚠️ AMP disabled (not supported on MPS/CPU).")

from torch.amp import autocast

# ---------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------
latent_dim = 100
batch_size = 64
epochs = 50
sample_folder = "samples2"
os.makedirs(sample_folder, exist_ok=True)

# ---------------------------------------------------------
# DATASET (MNIST)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------
# GENERATOR
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, 1, 28, 28)


# ---------------------------------------------------------
# DISCRIMINATOR
# ---------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

# ---------------------------------------------------------
# LOSS + OPTIMIZERS
# ---------------------------------------------------------
criterion = nn.BCELoss()
opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ---------------------------------------------------------
# SAVE SAMPLES
# ---------------------------------------------------------
def save_samples(epoch):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(25, latent_dim).to(device)
        fake_imgs = generator(z).cpu()

    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    idx = 0
    for i in range(5):
        for j in range(5):
            axes[i, j].imshow(fake_imgs[idx].squeeze(), cmap="gray")
            axes[i, j].axis("off")
            idx += 1

    plt.tight_layout()
    plt.savefig(f"{sample_folder}/epoch_{epoch}.png")
    plt.close()
    generator.train()


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
print("\nStarting Training...\n")

for epoch in range(1, epochs + 1):
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", colour="magenta")

    for real_imgs, _ in pbar:
        real_imgs = real_imgs.to(device)
        batch = real_imgs.size(0)

        real_labels = torch.ones(batch, 1).to(device)
        fake_labels = torch.zeros(batch, 1).to(device)

        # -------------------------
        # TRAIN DISCRIMINATOR
        # -------------------------
        z = torch.randn(batch, latent_dim).to(device)
        fake_imgs = generator(z)

        if use_amp:
            with autocast(device_type="cuda"):
                real_output = discriminator(real_imgs)
                fake_output = discriminator(fake_imgs.detach())
                d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
        else:
            real_output = discriminator(real_imgs)
            fake_output = discriminator(fake_imgs.detach())
            d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # -------------------------
        # TRAIN GENERATOR
        # -------------------------
        if use_amp:
            with autocast(device_type="cuda"):
                fake_output = discriminator(fake_imgs)
                g_loss = criterion(fake_output, real_labels)
        else:
            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        pbar.set_postfix(G=float(g_loss), D=float(d_loss))

    # Save sample every 10 epochs
    if epoch % 10 == 0:
        save_samples(epoch)

print("Training complete!")
