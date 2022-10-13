from args import *
from data import LiverLoader, DataLoader
from seg_train import train_segmentation_model, train_segmentation_model_with_backdoor
from attack_train import train_attack_model


def get_victim(data, args):
    victim_train = LiverLoader(
        data.victim_train_paths, backdoor_train=True,
        trigger_type=args.triggertype, trigger_size=args.triggersize, trigger_val=args.triggerval, trigger_prob=args.poison,
    )
    victim_train_dataloader = DataLoader(victim_train, batch_size=int(SEG_BATCH_SIZE), shuffle=True)
    
    victim_val = LiverLoader(data.victim_val_paths)
    victim_val_dataloader = DataLoader(victim_val, batch_size=int(SEG_BATCH_SIZE))

    victim_val_backdoored = LiverLoader(
        data.victim_val_paths, backdoor_test=True,
        trigger_type=args.triggertype, trigger_size=args.triggersize, trigger_val=args.triggerval, trigger_prob=args.poison,
    )
    victim_val_backdoored_dataloader = DataLoader(victim_val_backdoored, batch_size=int(SEG_BATCH_SIZE))

    victim_model = train_segmentation_model_with_backdoor(
        args.victim, victim_train_dataloader, victim_val_dataloader, victim_val_backdoored_dataloader, SEG_EPOCHS, SEG_LR)
    
    return victim_model


def get_shadow(data, args):
    shadow_train = LiverLoader(data.shadow_train_paths)
    shadow_train_dataloader = DataLoader(shadow_train, batch_size=int(SEG_BATCH_SIZE), shuffle=True)
    shadow_val = LiverLoader(data.shadow_val_paths)
    shadow_val_dataloader = DataLoader(shadow_val, batch_size=int(SEG_BATCH_SIZE))

    shadow_model = train_segmentation_model(args.shadow, shadow_train_dataloader, shadow_val_dataloader, SEG_EPOCHS, SEG_LR)

    return shadow_model


def get_attack(data, args, victim_model, shadow_model):
    attack_train = LiverLoader(data.shadow_attack_paths, attack=True)
    attack_train_dataloader = DataLoader(attack_train, batch_size=ATTACK_BATCH_SIZE, shuffle=True)
    attack_val = LiverLoader(data.victim_attack_paths, attack=True)
    attack_val_dataloader = DataLoader(attack_val, batch_size=ATTACK_BATCH_SIZE)

    attack_model = train_attack_model(
        args, shadow_model, victim_model, attack_train_dataloader, attack_val_dataloader, ATTACK_EPOCHS, ATTACK_LR)

    return attack_model
