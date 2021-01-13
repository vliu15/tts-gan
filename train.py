# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" Contains entry point for running training. """

import argparse
from datetime import datetime
from hydra.utils import instantiate
import math
import numpy as np
from omegaconf import OmegaConf
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from dataset import AudioDataset
from modules.utils import plot_spectrogram_to_numpy, seed_everything, weights_init
from trainer import Trainer


def parse_arguments():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser()

    # Model.
    parser.add_argument("--config", type=str, default="config.yml", help="Path to model config.")
    parser.add_argument("--resume", type=str, default=None, help="Path to model checkpoint.")

    # Dataloader & dataset.
    parser.add_argument("--metadata_file", type=str, default="/home/vliu15/LJSpeech-1.1/metadata.csv", help="Path to metadata.csv.")
    parser.add_argument("--cmudict_file", type=str, default="/home/vliu15/cmu_dictionary", help="Path to CMU phoneme dictionary.")

    parser.add_argument("--train_files", type=str, default="train_files.txt", help="Path to list of train files.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--train_segment_length", type=int, default=18000, help="Audio segment length for training.")

    parser.add_argument("--val_files", type=str, default="test_files.txt", help="Path to list of validation files.")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Batch size for validation.")
    parser.add_argument("--val_segment_length", type=int, default=36000, help="Audio segment length for validation.")

    # Optimizer & scheduler.
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay value.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")

    # Loss weights.
    parser.add_argument("--l_nll", type=float, default=1.0, help="Weight of nll latent loss.")
    parser.add_argument("--l_mse", type=float, default=0.1, help="Weight of mse length loss.")
    parser.add_argument("--l_reg", type=float, default=1e-4, help="Weight of orthogonal regularization.")

    # Train parameters.
    parser.add_argument("--log_dir", type=str, default="/home/vliu15/tts-gan/logs", help="Where to log all training outputs.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Norm of gradients to clip to.")

    return parser.parse_args()


def train(trainer, epochs, dataloaders, optimizers, schedulers, loss_weights, log_dir, device, max_grad_norm: float = 1.0):
    """ Trains model fully. """
    # Unpack.
    start_epoch, end_epoch = epochs
    train_dataloader, val_dataloader = dataloaders
    d_optimizer, g_optimizer = optimizers
    d_scheduler, g_scheduler = schedulers

    global_step = 0
    writer = SummaryWriter(log_dir)
    version = os.path.basename(log_dir)

    for i in range(start_epoch, end_epoch):

        # For weighting hard and soft spectrogram prediction loss.
        alpha = math.exp(-i / math.sqrt(end_epoch))
        loss_weights["hard"] = alpha
        loss_weights["soft"] = 1. - alpha

        trainer.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader), mininterval=1, desc="[version={},epoch={}]".format(version, i))
        for batch_idx, batch in enumerate(pbar):
            batch = [example.to(device) for example in batch]

            # Discriminator.
            d_optimizer.zero_grad()
            d_loss_dict = trainer.d_step(*batch, jitter_steps=60, debug=(batch_idx % 50 == 0))
            d_loss = d_loss_dict["real"] + d_loss_dict["fake"]
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.discriminator_parameters, max_grad_norm)
            d_optimizer.step()

            # Generator.
            g_optimizer.zero_grad()
            g_loss_dict = trainer.g_step(*batch, jitter_steps=60, debug=(batch_idx % 50 == 0))
            g_loss = 0.0
            for name, loss in g_loss_dict.items():
                weight = loss_weights.get(name, 1.0)
                g_loss = g_loss + weight * loss
            g_loss.backward()
            trainer.apply_orthogonal_regularization(trainer.generator_parameters, weight=loss_weights["reg"])
            torch.nn.utils.clip_grad_norm_(trainer.generator_parameters, max_grad_norm)
            g_optimizer.step()

            # Log training losses.
            pbar.set_description("[version={},epoch={}] d: {}, g: {}".format(version, i, round(d_loss.item(), 4), round(g_loss.item(), 4)))
            global_step += 1
            writer.add_scalar("train_d_loss", d_loss.item(), global_step)
            writer.add_scalar("train_g_loss", g_loss.item(), global_step)

        # Post epoch management.
        pbar.close()
        d_scheduler.step()
        g_scheduler.step()
        if (i + 1) % 5 == 0:
            torch.save({
                "epoch": i,
                "trainer": trainer.state_dict(),
                "d_optim": d_optimizer.state_dict(),
                "g_optim": g_optimizer.state_dict(),
                "d_sched": d_scheduler.state_dict(),
                "g_sched": g_scheduler.state_dict(),
            }, os.path.join(log_dir, "model_{}.pt".format(i)))

        trainer.eval()
        with torch.no_grad():
            d_loss = 0.0
            g_loss = 0.0
            for batch_idx, batch in enumerate(val_dataloader):
                batch = [example.to(device) for example in batch]
                d_loss_dict, g_loss_dict, y, y_pred, z, z_pred = trainer.step(*batch, jitter_steps=60, debug=True)

                # Accumulate discriminator loss.
                d_loss += d_loss_dict["real"].item() + d_loss_dict["fake"].item()

                # Accumulate generator loss.
                for name, loss in g_loss_dict.items():
                    weight = loss_weights.get(name, 1.0)
                    g_loss += weight * loss.item()

                # Log validation images of first validation example.
                if batch_idx == 0:
                    writer.add_image("val_mel_target", plot_spectrogram_to_numpy(z[0].cpu().numpy()), global_step, dataformats="HWC")
                    writer.add_image("val_mel_predicted", plot_spectrogram_to_numpy(z_pred[0].cpu().numpy()), global_step, dataformats="HWC")
                    writer.add_audio("val_audio_target", np.nan_to_num(y[0].cpu().numpy()), global_step, sample_rate=trainer.sampling_rate)
                    writer.add_audio("val_audio_predicted", np.nan_to_num(y_pred[0].cpu().numpy()), global_step, sample_rate=trainer.sampling_rate)

            d_loss /= len(val_dataloader)
            g_loss /= len(val_dataloader)

            # Log validation losses.
            writer.add_scalar("val_d_loss", d_loss, global_step)
            writer.add_scalar("val_g_loss", g_loss, global_step)
            print("[validation,epoch={}] d: {}, g: {}".format(global_step, round(d_loss, 4), round(g_loss, 4)))


def main():
    """ Entry point to training function. """
    seed_everything(1234)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.enabled = True

    args = parse_arguments()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    version = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, version)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    os.makedirs(log_dir, 0o777)

    # Set constants.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    length_scale = int(np.prod(config.trainer.audio_generator.decoder_scales))
    assert args.train_segment_length % length_scale == 0
    assert args.val_segment_length % length_scale == 0

    # Instantiate trainer.
    trainer = Trainer(**config.trainer)
    trainer = trainer.to(device)
    trainer.apply(weights_init)
    trainer.print_model_summary()

    # Instantiate dataset and dataloader.
    train_dataloader = torch.utils.data.DataLoader(
        AudioDataset(
            args.train_files,
            args.metadata_file,
            args.cmudict_file,
            args.train_segment_length,
            length_scale,
            config.trainer.sampling_rate,
            mu_law=config.trainer.mu_law,
        ),
        collate_fn=AudioDataset.collate_fn,
        batch_size=args.train_batch_size,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=20,
    )
    val_dataloader = torch.utils.data.DataLoader(
        AudioDataset(
            args.val_files,
            args.metadata_file,
            args.cmudict_file,
            args.val_segment_length,
            length_scale,
            config.trainer.sampling_rate,
            mu_law=config.trainer.mu_law,
        ),
        collate_fn=AudioDataset.collate_fn,
        batch_size=args.val_batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=8,
    )

    # Instantiate optimizers and schedulers.
    d_optimizer = torch.optim.AdamW(trainer.discriminator_parameters, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=args.epochs)
    g_optimizer = torch.optim.AdamW(trainer.generator_parameters, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=args.epochs)

    # Load checkpoint if specified.
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        trainer.load_state_dict(checkpoint["trainer"])
        d_optimizer.load_state_dict(checkpoint["d_optim"])
        g_optimizer.load_state_dict(checkpoint["g_optim"])
        d_scheduler.load_state_dict(checkpoint["d_sched"])
        g_scheduler.load_state_dict(checkpoint["g_sched"])

    # Train.
    train(
        trainer,
        (start_epoch, args.epochs),
        (train_dataloader, val_dataloader),
        (d_optimizer, g_optimizer),
        (d_scheduler, g_scheduler),
        {"nll": args.l_nll, "mse": args.l_mse, "reg": args.l_reg},
        log_dir,
        device,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()
