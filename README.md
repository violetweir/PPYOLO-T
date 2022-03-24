<<<<<<< HEAD

### We choose Training Data as the sample for training the network, and choose Test B as the validation set to test the ability of our networkk ：[Dataset Link](https://onedrive.live.com/?authkey=%21APblhWGPHFPVZ5Q&id=6A8143A3173E5D5A%211074&cid=6A8143A3173E5D5A)


### Relevant weights are still being sorted out 

### After you download the relevant data, put the training set into ./data/Train and change the name of the photo folder to JPEGImages 

### After you download the relevant data, put the Testb set into ./data/Test and change the name of the photo folder to JPEGImages 




### If you want to retrain, execute the following code

```
export CUDA_VISIBLE_DEVICES=0 
python tools/train.py -c configs/ppyolo/ppyolov2_Bot_50_Xhead_640.yml 
```

### If you have multiple GPUs, execute the following code

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyolo/ppyolov2_Bot_50_Xhead_640.yml
```


### If you want to perform an evaluation, you can execute
```
export CUDA_VISIBLE_DEVICES=0 
python tools/eval.py -c configs/ppyolo/ppyolov2_Bot_50_Xhead_640.yml -o weights={path of Weights}
```


### We use PaddleDetection as the basic framework to build [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
