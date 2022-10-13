import numpy as np
import torch
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_segmentation_model(encoder, dataloader, val_dataloader, epochs, lr):

    model = smp.Unet(encoder_name=encoder, in_channels=1, classes=1).to(device)

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


def train_segmentation_model_with_backdoor(encoder, dataloader, val_dataloader, val_backdoor_dataloader, epochs, lr):

    model = smp.Unet(encoder_name=encoder, in_channels=1, classes=1).to(device)

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

        print('Training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4))

        if epoch % 10 == 0: 
            validate_segmentation_model(model, val_dataloader)
            validate_segmentation_model(model, val_backdoor_dataloader, backdoor_test=True)

    # test one final time after the training is finished
    validate_segmentation_model(model, val_dataloader)
    validate_segmentation_model(model, val_backdoor_dataloader, backdoor_test=True)

    return model


def validate_segmentation_model(model, dataloader, backdoor_test=False):
    criterion = smp.losses.DiceLoss('binary')

    val_loss_data = []

    mask_sum = 0
    lbl_sum = 0
    
    model.eval()
    with torch.no_grad():
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)
            pred = model(img)

            loss = criterion(pred.float(), lbl.float())
            val_loss_data.append(loss.item())

            pred_clipped = torch.clip(pred, min=0., max=1.)
            mask_sum += torch.sum(pred_clipped)
            lbl_sum += torch.sum(lbl)

    val_loss = np.sum(np.array(val_loss_data))/len(val_loss_data)

    total = len(dataloader.dataset)

    if backdoor_test:
        print('Backdoored validation loss:', round(val_loss,4))
        print('  Average output mask sum:', mask_sum.item()/total)
        print('  Average true musk sum:', lbl_sum.item()/total)
    else:
        print('Non-backdoored validation loss:', round(val_loss,4))
        print('  Average output mask sum:', mask_sum.item()/total)
        print('  Average true musk sum:', lbl_sum.item()/total)

    return val_loss