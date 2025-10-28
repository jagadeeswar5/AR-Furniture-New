import gradio as gr
import requests
import json
import base64
from PIL import Image
import io
import os

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed URL

def upload_and_process_image(image, text_weight, visual_weight, color_boost):
    """Upload image and get furniture recommendations"""
    try:
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Prepare files and data
        files = {"file": ("image.png", img_bytes, "image/png")}
        data = {
            "text_weight": text_weight,
            "visual_weight": visual_weight,
            "color_boost": color_boost
        }
        
        # Make API call to your FastAPI backend
        response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("recommendations", []), result.get("message", "Success!")
        else:
            return [], f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return [], f"Error: {str(e)}"

def chat_with_bot(message, history):
    """Chat with the AI furniture assistant"""
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json={"message": message})
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Sorry, I couldn't process that.")
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def segment_image(image, mask_data):
    """Segment image using SAM"""
    try:
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Prepare files and data
        files = {"file": ("image.png", img_bytes, "image/png")}
        data = {"mask_data": json.dumps(mask_data)}
        
        # Make API call
        response = requests.post(f"{API_BASE_URL}/segment", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            # Convert base64 image back to PIL
            if "segmented_image" in result:
                img_data = base64.b64decode(result["segmented_image"])
                return Image.open(io.BytesIO(img_data))
        return image  # Return original if error
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return image

# Create Gradio Interface
with gr.Blocks(title="AR Furniture App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AR Furniture App")
    gr.Markdown("Upload a room image and get AI-powered furniture recommendations!")
    
    with gr.Tab("Furniture Recommendations"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Room Image",
                    type="pil",
                    height=400
                )
                
                with gr.Row():
                    text_weight = gr.Slider(
                        minimum=0, maximum=100, value=75,
                        label="Text Preference Weight (%)"
                    )
                    visual_weight = gr.Slider(
                        minimum=0, maximum=100, value=25,
                        label="Visual Analysis Weight (%)"
                    )
                
                color_boost = gr.Checkbox(
                    label="Enable Color Boost",
                    value=False
                )
                
                recommend_btn = gr.Button("Get Recommendations", variant="primary")
            
            with gr.Column():
                recommendations_output = gr.JSON(
                    label="Furniture Recommendations"
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        recommend_btn.click(
            fn=upload_and_process_image,
            inputs=[image_input, text_weight, visual_weight, color_boost],
            outputs=[recommendations_output, status_output]
        )
    
    with gr.Tab("AI Chatbot"):
        chatbot = gr.Chatbot(
            label="Furniture Assistant",
            height=400
        )
        msg = gr.Textbox(
            label="Ask me about furniture!",
            placeholder="What kind of sofa would work in my living room?"
        )
        msg.submit(
            fn=chat_with_bot,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
    
    with gr.Tab("Image Segmentation"):
        with gr.Row():
            with gr.Column():
                seg_image_input = gr.Image(
                    label="Upload Image for Segmentation",
                    type="pil",
                    height=400
                )
                segment_btn = gr.Button("Segment Image", variant="primary")
            
            with gr.Column():
                segmented_output = gr.Image(
                    label="Segmented Result",
                    height=400
                )
        
        segment_btn.click(
            fn=segment_image,
            inputs=[seg_image_input, gr.State({})],
            outputs=[segmented_output]
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## 🏠 AR Furniture App
        
        This application uses advanced AI to help you design your living space:
        
        - **AI-Powered Recommendations**: Get personalized furniture suggestions
        - **Image Segmentation**: Use SAM to identify objects in your room
        - **Interactive Chatbot**: Get design advice from an AI assistant
        - **AR Visualization**: See how furniture looks in your space
        
        ### Features:
        - Upload room images
        - Get furniture recommendations based on visual analysis
        - Chat with AI assistant for design advice
        - Segment images to identify objects
        - Customizable recommendation weights
        
        ### Technologies Used:
        - FastAPI backend
        - SAM (Segment Anything Model)
        - ResNet50 for visual features
        - OpenAI for natural language processing
        - Gradio for interactive interface
        """)

if __name__ == "__main__":
    # For local development
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
