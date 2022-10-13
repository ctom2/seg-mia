import argparse
from data import KvasirDataset
from utils_train import get_victim, get_shadow, get_attack
from global_attack import global_attack 


def main(args):
    # read all the paths from the liver folder
    data = KvasirDataset(args.trainsize)

    # train victim and shadow models
    print(" ** TRAINING VICTIM MODEL **")
    victim_model, _ = get_victim(data, args)
    print(" ** TRAINING SHADOW MODEL **")
    shadow_model, shadow_threshold = get_shadow(data, args)

    print(" ** MEMBERSHIP INFERENCE ATTACK **")

    print(' ** Type-I **')
    args.attacktype = 1
    _ = get_attack(data, args, victim_model, shadow_model)

    print(' ** Type-II **')
    args.attacktype = 2
    _ = get_attack(data, args, victim_model, shadow_model)

    print(' ** Global loss **')
    global_attack(data, args, victim_model, shadow_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # resnets, mobilenet_v2, vgg11
    parser.add_argument("--victim", type=str, default='resnet34') 
    # resnets, mobilenet_v2, vgg11
    parser.add_argument("--shadow", type=str, default='resnet34')
    # 1 -- Type-I attack, 
    # 2 -- Type-II attack,
    # 3 -- Global loss-based attack
    parser.add_argument("--attacktype", type=int, default=2)
    # 1 -- no defense,
    # 2 -- argmax defense,
    # 3 -- crop training,
    # 4 -- mix-up,
    # 5 -- min-max,
    parser.add_argument("--defensetype", type=int, default=0)
    # same for victim and shadow models (300 max)
    parser.add_argument("--trainsize", type=int, default=300)
    args = parser.parse_args()

    print("Victim encoder: {}".format(args.victim))
    print("Shadow encoder: {}".format(args.shadow))
    defenses = ['No defense', 'Argmax', 'Crop-training', 'Mix-up', 'Min-max', 'DP']
    print("Defense type: {}".format(defenses[args.defensetype - 1]))
    print("Train size: {}".format(args.trainsize))
    print()

    main(args)