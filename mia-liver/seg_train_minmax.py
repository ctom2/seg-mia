import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import segmentation_models_pytorch as smp
from torchvision import models
from args import LAMBDA
from seg_train import validate_segmentation_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def update_reg_model(
    reg_model,
    reg_lr,
    reg_epochs,
    reg_train_dataloader,
    seg_model,
):
    print(' -- Updating adversary regularisation model --')

    opt = torch.optim.NAdam(reg_model.parameters(), lr=reg_lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(reg_epochs):

        pred_labels = np.array([])
        true_labels = np.array([])

        reg_model.train()

        for data, labels, targets in reg_train_dataloader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)

            opt.zero_grad()
            with torch.no_grad():
                pred = seg_model(data)
            
            # running 2-channel attack by default
            cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
            s_output = torch.concat((pred, cat), dim=1)
            
            output = reg_model(s_output)

            loss = criterion(output.float(), targets.float().view(len(targets),1))
            loss.backward()
            opt.step()

            pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
            true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

            pred_labels = np.concatenate((pred_labels, pred_l))
            true_labels = np.concatenate((true_labels, true_l))

        print(
            'Training accuracy:', round(accuracy_score(true_labels, pred_labels),4),
            ', F-score:', round(f1_score(true_labels, pred_labels),4),
        )

    return reg_model


def train_segmentation_model_min_max(
    args,
    # required for the segmentation model 
    seg_train_dataloader, 
    seg_val_dataloader, 
    seg_epochs, 
    seg_lr,
    # required for the regularisation model
    reg_train_dataloader,
    reg_epochs,
    reg_lr,
):

    # adversarial regularisation done with Type-II attack
    ATTACK_INPUT_CHANNELS = 2

    reg_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    reg_model.conv1 = nn.Sequential(nn.Conv2d(ATTACK_INPUT_CHANNELS, 3, 1), reg_model.conv1,)
    reg_model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    reg_model.to(device)

    seg_model = smp.Unet(encoder_name=args.victim, in_channels=1, classes=1).to(device)

    criterion = smp.losses.DiceLoss('binary')
    opt = torch.optim.NAdam(seg_model.parameters(), lr=seg_lr, betas=(0.9, 0.999))

    for epoch in range(seg_epochs):
        print(' -- Staring training epoch {} --'.format(epoch + 1))

        # ADVERSARY REGULARISATION MODEL UPDATE
        reg_model = update_reg_model(reg_model, reg_lr, reg_epochs, reg_train_dataloader, seg_model)

        # SEGMENTATION MODEL UPDATE
        print(' -- Updating segmentation model --')

        seg_model.train()
        train_loss_data = []
        seg_loss_data = []
        reg_loss_data = []

        for img, lbl in seg_train_dataloader:
            img, lbl = img.to(device), lbl.to(device)

            opt.zero_grad()
            pred = seg_model(img)
            loss = criterion(pred.float(), lbl.float())
            seg_loss_data.append(loss.item())

            # getting outputs of the regularisation model on the training batch
            cat = lbl.view(img.shape[0],1,img.shape[2],img.shape[3])
            s_output = torch.concat((pred, cat), dim=1)
            reg_pred = reg_model(s_output)

            reg_loss = torch.sum(reg_pred)
            reg_loss_data.append(reg_loss.cpu().item()/img.shape[0])

            # summing the losses
            loss = loss + reg_loss * LAMBDA

            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        train_loss = np.sum(np.array(train_loss_data))/len(train_loss_data)
        print('Training loss:', round(train_loss,4))
        print('Seg. model loss: {}'.format(round(np.sum(np.array(seg_loss_data))/len(seg_loss_data),4)))
        print('Reg. model accuracy: {}'.format(round(np.sum(np.array(reg_loss_data))/len(reg_loss_data),4)))

        if epoch % 10 == 0: 
            validate_segmentation_model(seg_model, seg_val_dataloader)

    validate_segmentation_model(seg_model, seg_val_dataloader)

    return seg_model, train_loss