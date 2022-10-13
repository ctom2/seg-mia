import numpy as np
import torch
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_segmentation_model(encoder, dataloader, val_dataloader, epochs, lr):

    model = smp.Unet(encoder_name=encoder, in_channels=3, classes=1).to(device)

    criterion = smp.losses.DiceLoss('binary')
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred.float(), lbl.float())
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        train_loss = np.sum(np.array(train_loss_data))/len(train_loss_data)
        print('Training loss:', round(train_loss,4))

        if epoch % 10 == 0: validate_segmentation_model(model, val_dataloader)

    val_loss = validate_segmentation_model(model, val_dataloader)

    return model, train_loss


def validate_segmentation_model(model, dataloader):
    criterion = smp.losses.DiceLoss('binary')

    val_loss_data = []
    
    model.eval()
    with torch.no_grad():
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)
            pred = model(img)

            loss = criterion(pred.float(), lbl.float())
            val_loss_data.append(loss.item())

    val_loss = np.sum(np.array(val_loss_data))/len(val_loss_data)

    # Validation results
    print('Validation loss:', round(val_loss,4))
    
    return val_loss