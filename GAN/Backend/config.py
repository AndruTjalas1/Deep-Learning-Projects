"""Configuration management for DCGAN training."""

from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999


class ImageConfig(BaseModel):
    """Image settings."""
    resolution: int = 64
    channels: int = 3


class GeneratorConfig(BaseModel):
    """Generator architecture settings."""
    latent_dim: int = 100
    feature_maps: int = 64


class DiscriminatorConfig(BaseModel):
    """Discriminator architecture settings."""
    feature_maps: int = 64


class SamplingConfig(BaseModel):
    """Sample generation settings."""
    sample_interval: int = 100
    num_samples: int = 16


class DataConfig(BaseModel):
    """Data loading settings."""
    dataset_path: str = "./data"
    train_split: float = 0.8
    num_workers: int = 4


class OutputConfig(BaseModel):
    """Output directory settings."""
    samples_dir: str = "./samples"
    models_dir: str = "./saved_models"
    logs_dir: str = "./logs"


class DeviceConfig(BaseModel):
    """Device settings."""
    use_gpu: bool = True


class Config(BaseSettings):
    """Complete configuration model."""
    training: TrainingConfig = TrainingConfig()
    image: ImageConfig = ImageConfig()
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    sampling: SamplingConfig = SamplingConfig()
    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()
    device: DeviceConfig = DeviceConfig()

    class Config:
        """Pydantic settings."""
        env_file = ".env"

    @classmethod
    def load_from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)

    def save_to_yaml(self, output_path: str = "config.yaml") -> None:
        """Save current configuration to YAML file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'training': self.training.model_dump(),
            'image': self.image.model_dump(),
            'generator': self.generator.model_dump(),
            'discriminator': self.discriminator.model_dump(),
            'sampling': self.sampling.model_dump(),
            'data': self.data.model_dump(),
            'output': self.output.model_dump(),
            'device': self.device.model_dump(),
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration instance."""
    if config_path:
        return Config.load_from_yaml(config_path)
    
    if Path("config.yaml").exists():
        return Config.load_from_yaml("config.yaml")
    
    return Config()
