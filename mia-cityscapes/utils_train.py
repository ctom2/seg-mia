from args import *
from data import CityscapesLoader, DataLoader
from seg_train import train_segmentation_model
from seg_train_crop import train_segmentation_model_crop
from seg_train_mix import train_segmentation_model_mix
from attack_train import train_attack_model


def get_victim(data, args):
    victim_train = CityscapesLoader(data.victim_train_paths)
    victim_train_dataloader = DataLoader(victim_train, batch_size=int(SEG_BATCH_SIZE), shuffle=True)
    victim_val = CityscapesLoader(data.victim_val_paths)
    victim_val_dataloader = DataLoader(victim_val, batch_size=int(SEG_BATCH_SIZE))

    # no defense or argmax defense
    if (args.defensetype == 1) or (args.defensetype == 2):
        victim_model = train_segmentation_model(args.victim, victim_train_dataloader, victim_val_dataloader, SEG_EPOCHS, SEG_LR)
    # crop training
    elif args.defensetype == 3:
        victim_model = train_segmentation_model_crop(args.victim, victim_train_dataloader, victim_val_dataloader, 3*SEG_EPOCHS, SEG_LR)
    # mix-up
    elif args.defensetype == 4:
        victim_model = train_segmentation_model_mix(args.victim, victim_train_dataloader, victim_val_dataloader, SEG_EPOCHS, SEG_LR)

    return victim_model


def get_shadow(data, args):
    shadow_train = CityscapesLoader(data.shadow_train_paths)
    shadow_train_dataloader = DataLoader(shadow_train, batch_size=int(SEG_BATCH_SIZE), shuffle=True)
    shadow_val = CityscapesLoader(data.shadow_val_paths)
    shadow_val_dataloader = DataLoader(shadow_val, batch_size=int(SEG_BATCH_SIZE))

    shadow_model = train_segmentation_model(args.shadow, shadow_train_dataloader, shadow_val_dataloader, SEG_EPOCHS, SEG_LR)

    return shadow_model


def get_attack(data, args, victim_model, shadow_model):
    attack_train = CityscapesLoader(data.shadow_attack_paths, attack=True)
    attack_train_dataloader = DataLoader(attack_train, batch_size=ATTACK_BATCH_SIZE, shuffle=True)
    attack_val = CityscapesLoader(data.victim_attack_paths, attack=True)
    attack_val_dataloader = DataLoader(attack_val, batch_size=ATTACK_BATCH_SIZE)

    attack_model = train_attack_model(
        args, shadow_model, victim_model, attack_train_dataloader, attack_val_dataloader, ATTACK_EPOCHS, ATTACK_LR)

    return attack_model
