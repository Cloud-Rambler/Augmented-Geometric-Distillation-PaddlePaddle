# Augmented-Geometric-Distillation-PaddlePaddle
PaddlePaddle code for Augmented Geometric Distillation

Pre-training models can be found [here](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md#ResNet)

Datasets Structure
```
./data
- market
  - bounding_box_test
  - bounding_box_train
  - query
 
- msmt17
  - bounding_box_test
  - bounding_box_train
  - query

...
```

To train the baisc model on task $ T_1 $ (MSMT17), then run:

```
python ./tools/main.py -g 0 --dataset msmt17 --logs-dir ./logs/msmt17
```

To generation dreaming data via DeepInversion [1], then run:
```
python ./tools/inversion.py -g 0 --generation-dir ./data/generations_r50_msmt17 --shots 40 --iters 640 --teacher ./logs/msmt17
```
To train the incremental model on task $ T_2 $ (Market) with Geometric Distillation loss, then run:

```
python ./tools/main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_GD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./ppcls/configs/res-triangle.yaml
```
To train the incremental model on task $ T_2 $ (Market) with simple Geometric Distillation loss (detailed in Supp. and usually report better performance), then run:
```

python ./tools/main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_simGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./ppcls/configs/sim-res-triangle.yaml
```

To train the incremental model on task $ T_2 $ (Market) with Augmented Distillation, then run:
```

python .tools/main_incrementalX.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_XsimGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --peers 2 --epoch 80 --algo-config ./ppcls/configs/inverXion.yaml
```
To evaluate the incremental model on task $ T_2 $ (Market) with Augmented Distillation, then run:

```angular2svg
python ./tools/evaluate.py --dataset msmt17 market --ckpt ./logs/msmt17-market_XsimGD/checkpoint.pdparams --output 
```


If you want to reproduce other results in our work, just modify ` algo-config `.

Best wishes 🌈

[1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

