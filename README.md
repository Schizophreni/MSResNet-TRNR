# TRNR
#### The pytorch implementation of "Task-driven Image Deraining and Denoising with a Few Images Based on Patch Analysis"

* An overview of the task-driven learning approach.

![img](D:\Education\deepModels\TRNR\feature_visualization\TRNR.png)

* Model architecture for image deraining and denoising. 

![img](D:\Education\deepModels\TRNR\feature_visualization\net.png)

## Requirements

```python
#Requirements
######### Python Libraries #########
1. Pytorch (>=1.9.0)
2. tqdm
3. opencv-python (cv2)
4. PIL
######### CUDA Environment #########
CUDA version >= 11.3
```

## Inference

Please refer to the testing files:

- `test_rain.py `: utilized to test model on **Synthetic deraining**  datasets.
- `test_noise.py`: utilized to test model on **denoising** datasets.
- `test_real.py`: utilized to test model on **Real-world deraining** datasets.

Codes below show examples of testing function call.

```python
python test_rain.py 
python test_noise.py
python test_real.py
```

For a success testing, you may need to modify the **testing dataset key arguments**, the **configuration for test-set**, and the **checkpoint path**. 

The **Pre-training checkpoints** will be uploaded soon.

- testing dataset key arguments (Take Rain100L in `test_rain.py` for an example)

  ```python
  Rain100L_test_kwargs = {
      'dataset': 'Rain100L-t', # dataset name as a key for configuration 
      'type': 'rain', 
      'clean_dir': '/home/rw/Public/datasets/derain/Rain100L/norain', # ground truth folder
      'noise_dir': '/home/rw/Public/datasets/derain/Rain100L/rain', # rainy image folder
  }
  ```

- configuration for test-set (A dictionary to set the test clean/rainy image format | `test_rain.py` as an example)

  This test format refers to [ReHEN](https://github.com/nnUyi/ReHEN)

  ```python
  dataset_config = {
  'Rain100L-t': ('norain-{:03d}.png', 'rain-{:03d}.png', 100, 1),
  'Rain100H-t': ('norain-{:03d}.png', 'rain-{:03d}.png', 100, 1),
  'Rain800-t': ('norain-{:03d}.png', 'rain-{:03d}.png', 100, 1),
  # (dataset_name: (clean_format, rain_format, tot_number, beginning index))
  }
  ```

- checkpoint path (load specific checkpoint  | `test_rain.py` as an example)

  ```python
  model.load_model(model_save_path="XXX")
  ```

## Training models with TRNR

## Prepare your own datasets

## Customize your own model

## Ablation Instruction

## Results



