from latent_transport.config import DemoConfig
from latent_transport.train import train_model
from latent_transport.eval import evaluate_checkpoint

if __name__ == '__main__':
    cfg = DemoConfig()
    ckpt = train_model(cfg)
    evaluate_checkpoint(cfg, ckpt)