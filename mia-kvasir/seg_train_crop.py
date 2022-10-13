import numpy as np
import torch
import segmentation_models_pytorch as smp
from args import CROP_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_crop(img, lbl, random=True, x=0, y=0):
    if random:
        x = np.random.randint(0, img.shape[2] - CROP_SIZE, size=1)[0]
        y = np.random.randint(0, img.shape[3] - CROP_SIZE, size=1)[0]

    img = img[:,:,x:x+CROP_SIZE, y:y+CROP_SIZE]
    lbl = lbl[:,x:x+CROP_SIZE, y:y+CROP_SIZE]

    return img, lbl
    

def crop_stiching(img, lbl, seg_model):
    # PREDICTED MASK CROPS STICHING
    pred_full = torch.zeros(lbl.shape).to(device)
    for x in range(0,img.shape[3],CROP_SIZE):
        for y in range(0,img.shape[3],CROP_SIZE):
            data_crop, _ = get_crop(img, lbl, random=False, x=x, y=y)
            pred_full[:,x:x+CROP_SIZE,y:y+CROP_SIZE] = seg_model(data_crop)[:,0,:,:]

    pred_full = pred_full.view(img.shape[0],1,img.shape[2],img.shape[3])

    return pred_full


def train_segmentation_model_crop(encoder, dataloader, val_dataloader, epochs, lr):

    model = smp.Unet(encoder_name=encoder, in_channels=3, classes=1).to(device)

    criterion = smp.losses.DiceLoss('binary')
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)

            img, lbl = get_crop(img, lbl)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred.float(), lbl.float())
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        train_loss = np.sum(np.array(train_loss_data))/len(train_loss_data)
        print('Training loss:', round(train_loss,4))

        if epoch % 10 == 0: validate_segmentation_model_crop(model, val_dataloader)

    val_loss = validate_segmentation_model_crop(model, val_dataloader)

    return model, train_loss


def validate_segmentation_model_crop(model, dataloader):
    criterion = smp.losses.DiceLoss('binary')

    val_loss_data = []
    
    model.eval()
    with torch.no_grad():
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)

            # PREDICTED MASK CROPS STICHING
            pred_full = crop_stiching(img, lbl, model)

            loss = criterion(pred_full.float(), lbl.float())
            val_loss_data.append(loss.item())

    val_loss = np.sum(np.array(val_loss_data))/len(val_loss_data)

    # Validation results
    print('Validation loss:', round(val_loss,4))
    
    return val_loss