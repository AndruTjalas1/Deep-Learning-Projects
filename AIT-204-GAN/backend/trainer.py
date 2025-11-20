import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator, weights_init
from PIL import Image
import base64
import io


class DCGANTrainer:
    def __init__(self, device='cpu', nz=100, ngf=64, ndf=64, nc=3, lr=0.0002, beta1=0.5):
        self.device = device
        self.nz = nz
        self.nc = nc

        self.netG = Generator(nz, ngf, nc).to(device)
        self.netD = Discriminator(nc, ndf).to(device)

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.criterion = nn.BCELoss()

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        self.is_training = False
        self.current_epoch = 0
        self.metrics = {"g_losses": [], "d_losses": [], "real_scores": [], "fake_scores": []}

    # ----------------------------------------------------
    # DATA LOADER (NOW SUPPORTS CIFAR-10 CLASS FILTER)
    # ----------------------------------------------------

    def get_dataloader(self, dataset_name="mnist", batch_size=64, num_workers=0, class_filter=None):

        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # ----- MNIST -----
        if dataset_name == "mnist":
            dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

        # ----- Fashion MNIST -----
        elif dataset_name == "fashion_mnist":
            dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)

        # ----- CIFAR-10 -----
        elif dataset_name == "cifar10":
            dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)

            if class_filter:
                class_filter = class_filter.lower()
                if class_filter not in dataset.class_to_idx:
                    raise ValueError(f"Invalid CIFAR-10 class: {class_filter}")

                target_idx = dataset.class_to_idx[class_filter]
                indices = [i for i, y in enumerate(dataset.targets) if y == target_idx]
                dataset = torch.utils.data.Subset(dataset, indices)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if str(self.device) == "cuda" else False
        )

    # ----------------------------------------------------
    # TRAINING STEP (D + G)
    # ----------------------------------------------------

    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real = torch.full((batch_size,), 1.0, device=self.device)
        fake = torch.full((batch_size,), 0.0, device=self.device)

        # ----- Train Discriminator -----
        self.netD.zero_grad()

        output = self.netD(real_images).view(-1)
        d_real = self.criterion(output, real)
        d_real.backward()

        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake_images = self.netG(noise)
        output = self.netD(fake_images.detach()).view(-1)
        d_fake = self.criterion(output, fake)
        d_fake.backward()

        self.optimizerD.step()

        # ----- Train Generator -----
        self.netG.zero_grad()

        output = self.netD(fake_images).view(-1)
        g_loss = self.criterion(output, real)
        g_loss.backward()
        self.optimizerG.step()

        return {
            "loss_d": (d_real + d_fake).item(),
            "loss_g": g_loss.item(),
            "real_score": d_real.item(),
            "fake_score": d_fake.item()
        }

    # ----------------------------------------------------
    # IMAGE GENERATION
    # ----------------------------------------------------

    def generate_images(self, num_images=64, noise=None):
        self.netG.eval()
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(num_images, self.nz, 1, 1, device=self.device)
            images = self.netG(noise)
        self.netG.train()
        return images

    def images_to_base64(self, images, nrow=8):
        from torchvision.utils import make_grid
        images = images * 0.5 + 0.5
        images = images.clamp(0, 1)

        grid = make_grid(images, nrow=nrow)
        ndarr = (grid * 255).byte().permute(1, 2, 0).cpu().numpy()

        img = Image.fromarray(ndarr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ----------------------------------------------------
    # CHECKPOINTING
    # ----------------------------------------------------

    def save_checkpoint(self, path="checkpoint.pth"):
        torch.save({
            "epoch": self.current_epoch,
            "netG": self.netG.state_dict(),
            "netD": self.netD.state_dict(),
            "optG": self.optimizerG.state_dict(),
            "optD": self.optimizerD.state_dict(),
            "metrics": self.metrics
        }, path)

    def load_checkpoint(self, path="checkpoint.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.current_epoch = ckpt["epoch"]
        self.netG.load_state_dict(ckpt["netG"])
        self.netD.load_state_dict(ckpt["netD"])
        self.optimizerG.load_state_dict(ckpt["optG"])
        self.optimizerD.load_state_dict(ckpt["optD"])
        self.metrics = ckpt["metrics"]
