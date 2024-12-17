```
export OUTPUT_DIR="output_model"
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
```


Running commands:
```
accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet_with_checkpoint.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=genecodes/sketchybusiness \
 --resolution=512 \
 --learning_rate=1e-5 \
 --dataloader_num_workers=36 \
 --caption_column="text" \
 --conditioning_sketch_column="sketch" \
 --proportion_empty_prompts=0.6 \
 --validation_image "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a1e2a138ca20767/0a1e2a138ca20767_0.png" "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a2f63fbe231a803/0a2f63fbe231a803_0.png" \
 --validation_sketch "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a1e2a138ca20767/0a1e2a138ca20767_3.png" "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a2f63fbe231a803/0a2f63fbe231a803_3.png" \
 --validation_prompt "Add a green and blue baby walker" "Add a woman in a gray sleeveless top and blue jeans, seated on the floor with a brush" \
 --train_batch_size=6 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --cache_dir="/bigdrive/huggingface" \
 --checkpointing_steps=500 \
 --seed=42 \
 --report_to=wandb \
 --use_8bit_adam \
 --gradient_checkpointing \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=8 \
 --controlnet_model_name_or_path=$OUTPUT_DIR
```
 --max_train_samples=40000
 --resume_from_checkpoint=checkpoint-1000

Original:
```
accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet_original.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --dataloader_num_workers=16 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --cache_dir="/bigdrive/huggingface" \
 --checkpointing_steps=10 \
 --seed=42 \
 --report_to=wandb \
 --max_train_samples=1000
```

MultiControlNet:
```
accelerate launch --mixed_precision="fp16" --multi_gpu train_multicontrolnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --dataloader_num_workers=16 \
 --conditioning_sketch_column="sketch" \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_sketch "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --cache_dir="/bigdrive/huggingface" \
 --checkpointing_steps=10 \
 --seed=42 \
 --report_to=wandb \
 --max_train_samples=1000
```






 <!-- --report_to=wandb -->

```
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --dataloader_num_workers=16 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --cache_dir="/bigdrive/huggingface"
```


For inference:
```
accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=genecodes/sketchybusiness \
 --resolution=512 \
 --learning_rate=1e-5 \
 --dataloader_num_workers=36 \
 --caption_column="text" \
 --conditioning_sketch_column="sketch" \
 --proportion_empty_prompts=0.6 \
 --validation_image "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a1e2a138ca20767/0a1e2a138ca20767_0.png" "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a2f63fbe231a803/0a2f63fbe231a803_0.png" \
 --validation_sketch "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a1e2a138ca20767/0a1e2a138ca20767_3.png" "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a2f63fbe231a803/0a2f63fbe231a803_3.png" \
 --validation_prompt "Add a green and blue baby walker" "Add a woman in a gray sleeveless top and blue jeans, seated on the floor with a brush" \
 --train_batch_size=6 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --cache_dir="/bigdrive/huggingface" \
 --checkpointing_steps=500 \
 --seed=42 \
 --report_to=wandb \
 --use_8bit_adam \
 --gradient_checkpointing \
 --gradient_accumulation_steps=4 \
 --controlnet_model_name_or_path=$OUTPUT_DIR \
 --max_train_samples=1
```

accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet.py  --pretrained_model_name_or_path=$MODEL_DIR  --output_dir=$OUTPUT_DIR  --dataset_name=genecodes/sketchybusiness  --resolution=512  --learning_rate=1e-5  --dataloader_num_workers=36  --caption_column="text"  --conditioning_sketch_column="sketch"  --proportion_empty_prompts=0.6  --validation_image "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a1e2a138ca20767/0a1e2a138ca20767_0.png" "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a2f63fbe231a803/0a2f63fbe231a803_0.png"  --validation_sketch "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a1e2a138ca20767/0a1e2a138ca20767_3.png" "/bigdrive/datasets/sketchy2pix/final-pix2pix/0a2f63fbe231a803/0a2f63fbe231a803_3.png"  --validation_prompt "Add a green and blue baby walker" "Add a woman in a gray sleeveless top and blue jeans, seated on the floor with a brush"  --train_batch_size=6  --mixed_precision="fp16"  --tracker_project_name="controlnet-demo"  --cache_dir="/bigdrive/huggingface"  --checkpointing_steps=500  --seed=42  --report_to=wandb  --use_8bit_adam  --gradient_checkpointing  --gradient_accumulation_steps=4  --controlnet_model_name_or_path=$OUTPUT_DIR