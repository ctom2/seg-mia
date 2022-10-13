SEG_BATCH_SIZE=8
SEG_EPOCHS=70
SEG_LR=1e-4

ATTACK_BATCH_SIZE=4
ATTACK_EPOCHS=30
ATTACK_LR=1e-4

CROP_SIZE=128 # for crop training defense

LAMBDA=0.005 # weight for adversarial regularisation
REG_EPOCHS=1 # number of epochs for optimising adversarial regularisation

EPSILON=8.5
MAX_GRAD_NORM=2.0

OUTPUT_CHANNELS=1 # 1 for binary segmentation (liver, Kvasir-SEG)