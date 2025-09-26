# ARA Pipeline

This repository provides a workflow for compressing and fine-tuning models using the Adaptive Rank Allocation (ARA) method.  
The process is divided into several clear steps, from calibration to compression, mask training, fine-tuning, and optional quantization.

---

## Steps

### 1. Collect Calibration Data
Calibration data is needed to guide the compression process.  
Run the following command to collect it:

```bash
python ARA.py --model $model --step 1 --save_path $save_path
```

### 2. Generate a Uniform Compression Model

Once calibration data is ready, generate a uniformly compressed model:

```bash
python ARA.py --model $model --profiling_mat_path $profiling_mat_path --ratio 1.2 --step 1 --save_path $save_path
```

⚠️Important: The ratio must be greater than 1. Otherwise, the original weight matrix cannot be preserved.

### 3. Modify Model Configuration

After generating the compressed model, edit the config.json file inside the saved directory.
Change the model_type field according to your model family:

For LLaMA-based models:
```
"model_type": "svd_llama"
```
For Qwen3-based models:
```
"model_type": "svd_qwen3"
```
This ensures the model loads with the correct architecture.

### 4. Train the Mask

Next, train the adaptive mask to allocate ranks effectively:
```bash
python ARA.py --ratio $target_ratio --model_path $compressed_model_from_step2 --step 2 --save_path $save_path
```

### 5. (Optional) Fine-tuning with LoRA

If further fine-tuning is required, apply LoRA:
```bash
python utils/LoRA.py --prune_model $model_from_step4 --num_epochs 2 --output_dir $output_dir
```

### 6. (Optional) Quantization

For efficient inference, you can quantize the compressed model:
```
python quant_llama.py --model_path $model_path --dataset c4 --wbits 4 --true-sequential --act-order --new-eval --save $save_path
```

## Acknowledgement

This repository is built upon [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM).