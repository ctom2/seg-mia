import segmentation_models_pytorch as smp
import torch
import numpy as np
from seg_train import validate_segmentation_model
from data import LiverLoader, DataLoader
from args import SEG_BATCH_SIZE, ATTACK_BATCH_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_reference_idxs(model, dataloader, threshold):
    criterion = smp.losses.DiceLoss('binary')
    
    reference_idxs = []

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            img, lbl = data
            img, lbl = img.to(device), lbl.to(device)
            pred = model(img)

            loss = criterion(pred.float(), lbl.float())

            if loss <= threshold: reference_idxs.append(idx)

    print('{}/{} data samples chosen for reference dataset'.format(len(reference_idxs), len(dataloader)))

    return reference_idxs


def make_protected_training_data(args, data, unprotected_model, threshold):
    pre_reference_loader = LiverLoader(data.pre_reference_paths)
    pre_reference_dataloader = DataLoader(pre_reference_loader, batch_size=1, shuffle=False)
    reference_idxs = get_reference_idxs(unprotected_model, pre_reference_dataloader, threshold)

    reference_paths = {
        'imgs': data.pre_reference_paths['imgs'][reference_idxs],
        'lbls': data.pre_reference_paths['lbls'][reference_idxs],
    }

    mask = np.ones(len(data.pre_reference_paths['imgs']), np.bool)
    mask[reference_idxs] = 0

    # for evaluating the attack on the protected model (not used with the current implementation)
    reference_val_paths = {
        'imgs': np.concatenate([data.pre_reference_paths['imgs'][reference_idxs], data.pre_reference_paths['imgs'][mask]]),
        'lbls': np.concatenate([data.pre_reference_paths['lbls'][reference_idxs], data.pre_reference_paths['lbls'][mask]]),
        'member': np.concatenate([np.ones((len(reference_idxs))), np.zeros((len(mask)))]),
    }

    reference_loader = LiverLoader(reference_paths)
    reference_dataloader = DataLoader(reference_loader, batch_size=int(SEG_BATCH_SIZE), shuffle=True)
    
    return reference_dataloader


def train_protected_model(encoder, unprotected_model, dataloader, val_dataloader, epochs, lr):

    print(" ** TRAINING PROTECTED VICTIM MODEL **")

    protected_model = smp.Unet(encoder_name=encoder, in_channels=1, classes=1).to(device)

    opt = torch.optim.NAdam(protected_model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = smp.losses.DiceLoss('binary')

    unprotected_model.eval()

    for epoch in range(epochs):
        protected_model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for data, _ in dataloader:
            data = data.to(device)

            with torch.no_grad():
                t = unprotected_model(data)
                t = torch.round(torch.clip(t, 0, 1))

            opt.zero_grad()
            pred = protected_model(data)
            loss = criterion(pred.float(), t.float())
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        train_loss = np.sum(np.array(train_loss_data))/len(train_loss_data)
        print('Training loss:', round(train_loss,4))

        if epoch % 10 == 0: validate_segmentation_model(protected_model, val_dataloader)

    validate_segmentation_model(protected_model, val_dataloader)

    return protected_model, train_loss