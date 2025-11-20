import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os

# =========================================================
# DEVICE
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# =========================================================
# GENERATOR
# =========================================================
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


# =========================================================
# DISCRIMINATOR (always outputs batch-size # of scalars)
# =========================================================
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1)


# =========================================================
# DATASET LOADER
# =========================================================
def load_dataset(name="cifar10", batch_size=64, class_filter=None):

    transform_rgb = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    full = datasets.CIFAR10("./data", train=True, download=True, transform=transform_rgb)

    if class_filter:
        class_names = [
            "airplane","automobile","bird","cat","deer",
            "dog","frog","horse","ship","truck"
        ]
        allowed = [i for i, name in enumerate(class_names) if name in class_filter]
        dataset = [(img, lbl) for img, lbl in full if lbl in allowed]
    else:
        dataset = full

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =========================================================
# TRAINING LOOP
# =========================================================
def train(dataset="cifar10", class_filter=["dog"], epochs=10, batch_size=64, nz=100):

    print("\nLoading DOG-ONLY dataset...")
    dataloader = load_dataset(dataset, batch_size, class_filter)
    print(f"Total batches: {len(dataloader)}")

    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    os.makedirs("samples", exist_ok=True)

    for epoch in range(1, epochs + 1):

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{epochs}",
            colour="#ff69b4",  
            ncols=100
        )

        for real, _ in progress:

            real = real.to(device)
            batch = real.size(0)

            real_label = torch.ones(batch, device=device)
            fake_label = torch.zeros(batch, device=device)

            # Train D
            netD.zero_grad()
            out_real = netD(real)
            loss_real = criterion(out_real, real_label)
            loss_real.backward()

            noise = torch.randn(batch, nz, 1, 1, device=device)
            fake = netG(noise)

            out_fake = netD(fake.detach())
            loss_fake = criterion(out_fake, fake_label)
            loss_fake.backward()

            optimizerD.step()

            # Train G
            netG.zero_grad()
            out = netD(fake)
            lossG = criterion(out, real_label)
            lossG.backward()
            optimizerG.step()

            progress.set_postfix({
                "D": f"{(loss_real + loss_fake).item():.3f}",
                "G": f"{lossG.item():.3f}"
            })

        if epoch % 2 == 0:
            with torch.no_grad():
                samples = netG(fixed_noise)
                samples = samples * 0.5 + 0.5
                save_image(samples, f"samples/dogs_epoch_{epoch}.png")
                print(f"Saved dog samples at epoch {epoch}")

    torch.save(netG.state_dict(), "dcgan_dogs_G.pth")
    torch.save(netD.state_dict(), "dcgan_dogs_D.pth")
    print("\nTraining complete! Models saved.")
    

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    train(
        dataset="cifar10",
        class_filter=["dog"], 
        epochs=400,
        batch_size=64,
    )
