from fastapi import FastAPI, File, UploadFile, WebSocket, Depends
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException
import os
import torch
from PIL import Image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import GaussianDiffusion
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from io import BytesIO

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# WebSocket for real-time updates
class WebSocketUpdate(WebSocket):
    async def on_connect(self, websocket):
        await websocket.accept()


def load_models(models_path):
    # Load the transmitter model
    xm = load_model('transmitter')
    xm.load_state_dict(torch.load(os.path.join(models_path, 'xm_model.pth')))
    xm.eval()

    # Load the text300M model
    text300M = load_model('text300M')
    text300M.load_state_dict(torch.load(os.path.join(models_path, 'text300M_model.pth')))
    text300M.eval()

    # Load diffusion configuration
    diffusion_params = torch.load(os.path.join(models_path, 'diffusion_model.pth'))
    diffusion = GaussianDiffusion(**diffusion_params)

    return xm, text300M, diffusion


async def generate_3d_model(img_path, xm, text300M, diffusion, cameras, render_mode, size, websocket):
    img = Image.open(img_path)

    latents = sample_latents(
        batch_size=4,
        model=text300M,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs=dict(images=[img] * 4),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    images = []
    for i, latent in enumerate(latents):
        frame_images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        images.extend(frame_images)

        # Send progress to WebSocket
        await websocket.send_text(f'{{"type": "progress", "progress": {i / 4 * 100}}}')

    # Convert to GIF
    gif_bytes = gif_widget(images)

    # Send generated image path to WebSocket
    await websocket.send_text(f'{{"type": "image", "image": "/get_generated_image"}}')

    return gif_bytes


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocketUpdate):
    while True:
        await websocket.receive_text()

@app.post("/generate_3d_model")
async def generate_3d_model_from_image(
    img: UploadFile = File(...),
    models_path: str = '/content/saved',
    websocket: WebSocketUpdate = None
):
    try:
        img_path = "/tmp/" + img.filename
        with open(img_path, "wb") as buffer:
            buffer.write(img.file.read())

        xm, text300M, diffusion = load_models(models_path)
        cameras = create_pan_cameras(128, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        render_mode = 'nerf'
        size = 128

        gif_bytes=generate_3d_model(img_path, xm, text300M, diffusion, cameras, render_mode, size, websocket)

        return StreamingResponse(BytesIO(gif_bytes), media_type="image/gif")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))