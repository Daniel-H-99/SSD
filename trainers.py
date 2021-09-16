import torch
import torch.nn as nn
import time
from utils import AverageMeter, ProgressMeter, accuracy

def ssl(
    model,
    device,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with self-supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, data in enumerate(dataloader):
        images, target = data[0], data[1].to(device)
        images = torch.cat([images[0], images[1]], dim=0).to(device)
        bsz = target.shape[0]

        # basic properties of training
        if i == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        # Todo: Complete training on a batch
        raise NotImplementedError

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
