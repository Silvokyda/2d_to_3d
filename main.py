from fastapi import FastAPI, File, UploadFile, WebSocket, Depends, HTTPException, WebSocketDisconnect
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif
from PIL import Image
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# WebSocket for real-time updates
class WebSocketUpdate(WebSocket):
    async def on_connect(self, websocket):
        await websocket.accept()

# Load the ShapEImg2ImgPipeline model
pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16, variant="fp16").to("cuda")

async def generate_3d_model(image_path, pipe, guidance_scale, num_inference_steps, frame_size, websocket):
    image = Image.open(image_path).resize((256, 256))

    images = pipe(
        image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        frame_size=frame_size,
    ).images

    # Export the resulting images to a GIF
    gif_path = export_to_gif(images, "generated_3d_image.gif")

    # Send generated image path to WebSocket
    await websocket.send_text(f'{{"type": "image", "image": "/get_generated_image"}}')

    return gif_path

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        while websocket.application_state != WebSocketState.DISCONNECTED:
            data = await websocket.receive_text()
            if data == "close":
                await websocket.close()
                break
            else:
                print(f"Received data: {data}")
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {e}")
    except Exception as e:
        print(f"Error in WebSocket endpoint: {e}")
        await websocket.close(code=1006)

@app.post("/generate_3d_model")
async def generate_3d_model_from_image(
    img: UploadFile = File(...),
    websocket: WebSocketUpdate = None
):
    try:
        # Save the uploaded file to a temporary directory
        img_path = f"temp_{img.filename}"
        with open(img_path, "wb") as buffer:
            buffer.write(img.file.read())

        guidance_scale = 3.0
        num_inference_steps = 64
        frame_size = 256

        gif_path = await generate_3d_model(img_path, pipe, guidance_scale, num_inference_steps, frame_size, websocket)

        # Return the generated GIF as a streaming response
        return StreamingResponse(open(gif_path, "rb"), media_type="image/gif")

    except Exception as e:
        print(f"Error in generate_3d_model_from_image endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))