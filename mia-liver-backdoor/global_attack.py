import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import segmentation_models_pytorch as smp
from data import LiverLoader, DataLoader
from args import ATTACK_BATCH_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def global_attack(data, args, victim_model, threshold):
    attack_val = LiverLoader(data.victim_attack_paths, attack=True)
    attack_val_dataloader = DataLoader(attack_val, batch_size=1)

    criterion = smp.losses.DiceLoss('binary')
    pred_labels = np.array([])
    true_labels = np.array([])

    victim_model.eval()
    for data, labels, targets in attack_val_dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        with torch.no_grad(): pred = victim_model(data)

        instance_loss = criterion(pred, labels).item()

        # if instance loss is greater that the average train loss, then it is classified as non-member
        if instance_loss > threshold:
            pred_l = np.array([0])
        else:
            pred_l = np.array([1])

        true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

        pred_labels = np.concatenate((pred_labels, pred_l))
        true_labels = np.concatenate((true_labels, true_l))

    print(
        'Validation accuracy:', round(accuracy_score(true_labels, pred_labels),4),
        ', F-score:', round(f1_score(true_labels, pred_labels),4),
    )

    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))