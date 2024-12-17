# from huggingface_hub import login
from diffusers import ControlNetModel


controlnet_image_path = "output_model_trained_on_8hrs/controlnet_image"
controlnet_sketch_path = "output_model_trained_on_8hrs/controlnet_sketch"

controlnet_image_model = ControlNetModel.from_pretrained(controlnet_image_path)
controlnet_sketch_model = ControlNetModel.from_pretrained(controlnet_sketch_path)

controlnet_image_model.push_to_hub("SketchyBusinessControlNet_image")
controlnet_sketch_model.push_to_hub("SketchyBusinessControlNet_sketch")
