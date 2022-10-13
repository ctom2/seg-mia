import numpy as np
import torch
import segmentation_models_pytorch as smp
from seg_train import validate_segmentation_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mixup_batch(imgs, lbls):
    split = int(np.random.beta(a=2,b=2,size=1)[0] * imgs.shape[3])

    p = torch.randperm(len(imgs))
    imgs1, lbls1 = imgs[p], lbls[p]
    imgs2, lbls2 = imgs, lbls

    imgs1[:,:,:,split:] = imgs2[:,:,:,split:]
    lbls1[:,:,split:] = lbls2[:,:,split:]

    return imgs1, lbls1


def train_segmentation_model_mix(encoder, dataloader, val_dataloader, epochs, lr):

    model = smp.Unet(encoder_name=encoder, in_channels=3, classes=19).to(device)

    criterion = smp.losses.DiceLoss('multiclass')
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:
            img, lbl = mixup_batch(img, lbl)
            img, lbl = img.to(device), lbl.to(device)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred.float(), lbl.type(torch.int64))
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        print('Training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4))

        if epoch % 10 == 0: validate_segmentation_model(model, val_dataloader)

    val_loss = validate_segmentation_model(model, val_dataloader)

    return model, val_loss