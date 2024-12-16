import gradio as gr
from PIL import Image

def get_dimensions(img):
    if img is None:
        return None, None
    return img.width, img.height

def create_blank_image(width, height):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    return img

def run_model(input_image, sketch_image, prompt):
    # Placeholder for model inference
    if input_image is None:
        return None
    width, height = input_image.size
    output_image = create_blank_image(width, height)
    return output_image

with gr.Blocks() as demo:
    gr.Markdown("## Image + Sketch + Prompt to Output Image")

    with gr.Row():
        input_image = gr.Image(label="Upload an image", type="pil")
        text_prompt = gr.Textbox(label="Text Prompt")

    # Initialize with a default canvas_size; will update once image is uploaded
    sketchpad = gr.Sketchpad(
        label="Draw your black-and-white sketch here", 
        type="pil",
        # Just pick a default size; will be updated dynamically
        canvas_size=(256, 256)
    )

    width_state = gr.State()
    height_state = gr.State()

    run_button = gr.Button("Run Model")
    output_image = gr.Image(label="Output Image", type="pil")

    def update_sketchpad(img):
        if img is not None:
            w, h = get_dimensions(img)
            # Update the sketchpad with the new canvas size
            return gr.update(canvas_size=(w, h)), w, h
        else:
            return gr.update(), None, None

    input_image.change(
        fn=update_sketchpad,
        inputs=[input_image],
        outputs=[sketchpad, width_state, height_state]
    )

    run_button.click(
        fn=run_model, 
        inputs=[input_image, sketchpad, text_prompt], 
        outputs=[output_image]
    )

demo.launch()
