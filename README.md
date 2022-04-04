**ODPVT**: Object Detection with Pooled Vision Transformer
========
## Requirements

* It is a good idea to have a CUDA-capable GPU and a correctly configured nvidia/CUDA setup
* A COCO-annotated dataset (or write your own dataset and add the tag to datasets/\_\_init\_\_.py)
* `pip install -r requirements.txt`

## Parameters

The main script, main.py, has a variety of parameters available that can be changed to suit your needs.  For example, you could change the backbone model by simply changing the url that fetches the checkpoint and change the constructor's backbone using the `--backbone` argument to match. You could completely skip the finetuning bit and retrain a ODPVT model from scratch by simply setting `pretrained = False` in the "Load a model" block. 

**ODPVT**: Object Detection with Pooled Vision Transformer
========
PyTorch training code and pretrained models for **ODPVT** .
We replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **39 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. 

![ODPVT](.github/DETR.png)

**What it is**. Unlike traditional computer vision techniques, ODPVT approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a POOLED Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, ODPVT reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

**About the code**. We believe that object detection should not be more difficult than classification,
and should not require complex libraries for training and inference.
ODPVT is very simple to implement and experiment with. 
Training code follows this idea - it is not a library,
but simply a [main.py](main.py) importing model and criterion
definitions with standard training loops.

# Model Zoo
We provide baseline DETR and DETR-DC5 models, and plan to include more in future.
AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images,
with torchscript transformer.




## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
A single epoch takes 28 minutes, so 300 epoch training
takes around 6 days on a single machine with 8 V100 cards.


## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:

