from args import OUTPUT_CHANNELS
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_attack_model(args, shadow_model, victim_model, dataloader, val_dataloader, epochs, lr):
    
    if args.attacktype == 1:
        ATTACK_INPUT_CHANNELS = OUTPUT_CHANNELS
    else:
        ATTACK_INPUT_CHANNELS = OUTPUT_CHANNELS * 2

    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.conv1 = nn.Sequential(nn.Conv2d(ATTACK_INPUT_CHANNELS, 3, 1), model.conv1,)
    model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    model.to(device)
    
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    q_accuracy = []
    q_f1 = []

    for epoch in range(epochs):

        pred_labels = np.array([])
        true_labels = np.array([])

        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))

        for data, labels, targets in dataloader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)

            opt.zero_grad()
            with torch.no_grad():
                pred = shadow_model(data)

            # argmax defense
            if args.defensetype == 2:
                pred = torch.round(pred)
            
            if ATTACK_INPUT_CHANNELS == 2:
                cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
                s_output = torch.concat((pred, cat), dim=1)
            else:
                s_output = pred

            output = model(s_output)

            loss = criterion(output.float(), targets.float().view(len(targets),1))
            loss.backward()
            opt.step()

            pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
            true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

            pred_labels = np.concatenate((pred_labels, pred_l))
            true_labels = np.concatenate((true_labels, true_l))

        print(
            'Training accuracy:', round(accuracy_score(true_labels, pred_labels),4),
            ', F1-score:', round(f1_score(true_labels, pred_labels),4),
        )

        res = test_attack_model(args, model, val_dataloader, victim_model=victim_model, input_channels=ATTACK_INPUT_CHANNELS)

        if epoch >= epochs - 10:
            q_accuracy.append(res[0])
            q_f1.append(res[1])

    print('\n\nLast 10 epochs testing averages: accuracy: {}, F1-score: {}'.format(
        round(np.mean(q_accuracy),4),
        round(np.mean(q_f1),4),
    ))

    return model


def test_attack_model(args, model, dataloader, shadow_model=None, victim_model=None, accuracy_only=False, input_channels=1):
    pred_labels = np.array([])
    true_labels = np.array([])
    pr_data = np.array([])

    # Testing loop
    model.eval()
    for data, labels, targets in dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        with torch.no_grad():
            if shadow_model == None:
                pred = victim_model(data)
            else:
                pred = shadow_model(data)

            # argmax defense
            if args.defensetype == 2:
                pred = torch.round(pred)

        if input_channels == 2:
            cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
            s_output = torch.concat((pred, cat), dim=1)
        else:
            s_output = pred

        output = model(s_output)

        pr = output.float().view(data.shape[0]).detach().cpu().numpy()
        pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
        true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

        pred_labels = np.concatenate((pred_labels, pred_l))
        true_labels = np.concatenate((true_labels, true_l))
        pr_data = np.concatenate((pr_data, pr))


    a_score = round(accuracy_score(true_labels, pred_labels),4)
    f_score = round(f1_score(true_labels, pred_labels),4)

    if accuracy_only:
        print('Validation accuracy:', accuracy_score)
    else:
        print(
            'Validation accuracy:', a_score,
            ', F1-score:', f_score,
        )

        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))

    return a_score, f_score