# Adversarial attacks on segmentation models

Implementation of membership inference attacks 🗡 on (poisoned 🧪) binary and multi-class segmentation models. Running the experiments requires [Citycapes](https://www.cityscapes-dataset.com) dataset [1], [Medical Segmentation Decathlon](http://medicaldecathlon.com) dataset (its subset of liver images) [2] and [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) dataset [3]. The implementation assumes attacks on a victim model with the use of a single shadow model.

🎓 The code was developed for my master's thesis in data engineering and analytics at Technische Universität München (TUM).

## Prerequisites

* Python 3.7 or higher
* PyTorch 1.12.1 or higher
* Torchvision 0.13 or higher
* Opacus 1.2 or higher
* segmentation_models_pytorch 0.3.0 or higher

## Data

Cityscapes data need to be organised as follows:

```
seg-mia
└───mia-cityscapes
│   └───cityscapes
│   │   └───leftImg8bit
│   │   │   └───train
│   │   └───gtFine
│   │   │   └───train
...
```

Medical Segmentation Decathlon data need to be organised as follows:

```
seg-mia
└───mia-liver(-backdoor)
│   └───liver
│   │   └───imgs
│   │   └───lbls
...
```

Kvasir-SEG data need to be organised as follows:

```
seg-mia
└───mia-kvasir
│   └───Kvasir-SEG
│   │   └───images
│   │   └───masks
...
```

## Evaluation

Each attack can be evaluated by running `python main.py` and controlled with the following command-line options.

General settings:
* Arguments `--victim` and `--shadow` define the encoder architectures for victim and shadow U-Net segmentation models, respectively (see [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch))
* `--defensetype` sets the defense methods applied to a victim model
* `--trainsize` defines the size of the training dataset for the victim and shadow models

Settings for MIAs on poisoned models:
* `--triggertype` controls the trigger shape (either 'line' or 'square')
* `--triggersize` sets the height for a line trigger or the edge size of a square trigger (in pixels)
* `--triggerval` sets the trigger values
* `--poison` defines the poisoning probability for each training data sample; in the range [0,1]

Defense types:

| Defense type      | Description             | Implementation                                                    |
| ----------------- | ----------------------- | ----------------------------------------------------------------- |
| 1                 | No defense              | `mia-liver`, `mia-kvasir`, `mia-cityscapes`, `mia-liver-backdoor` |
| 2                 | Argmax                  | `mia-liver`, `mia-kvasir`, `mia-cityscapes`                       |
| 3                 | Crop training           | `mia-liver`, `mia-kvasir`, `mia-cityscapes`                       |
| 4                 | Mix-up                  | `mia-liver`, `mia-kvasir`, `mia-cityscapes`                       |
| 5                 | Min-max                 | `mia-liver`, `mia-kvasir`                                         |
| 6                 | DP                      | `mia-liver`                                                       |
| 7                 | Knowledge distiallation | `mia-liver`                                                       |


## References 

[1] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. “The cityscapes dataset for semantic urban scene understanding.” In: *Proceedings of the IEEE conference on computer vision and pattern recognition.* 2016, pp. 3213–3223.

[2] A. L. Simpson, M. Antonelli, S. Bakas, M. Bilello, K. Farahani, B. Van Ginneken, A. Kopp-Schneider, B. A. Landman, G. Litjens, B. Menze, et al. “A large annotated medical image dataset for the development and evaluation of segmentation algorithms.” In: *arXiv preprint* arXiv:1902.09063 (2019).

[3] K. Pogorelov, K. R. Randel, C. Griwodz, S. L. Eskeland, T. de Lange, D. Johansen, C. Spampinato, D.-T. Dang-Nguyen, M. Lux, P. T. Schmidt, M. Riegler, and P. Halvorsen. “KVASIR: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection.” In: *Proceedings of the 8th ACM on Multimedia Systems Conference.* MMSys’17. Taipei, Taiwan: ACM, 2017, pp. 164–169. isbn: 978-1-4503-5002-0. doi: 10.1145/3083187. 3083212.
