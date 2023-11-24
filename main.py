from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import os
import torch
from PIL import Image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import GaussianDiffusion
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from io import BytesIO

app = FastAPI()

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

def generate_3d_model_from_image(img: UploadFile = File(...), models_path: str = '/content/saved'):
    try:
        # Save the uploaded image
        img_path = "/tmp/" + img.filename
        with open(img_path, "wb") as buffer:
            buffer.write(img.file.read())

        # Load the models
        xm, text300M, diffusion = load_models(models_path)

        # Load the input image
        img = Image.open(img_path)

        # Sample latents using the input image
        latents = sample_latents(
            batch_size=4,
            model=text300M,  # Use text300M as the main model
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

        render_mode = 'nerf'  # you can change this to 'stf'
        size = 128

        cameras = create_pan_cameras(size, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Convert to GIF
        images = []
        for i, latent in enumerate(latents):
            frame_images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            images.extend(frame_images)

        # Create a GIF
        gif_bytes = gif_widget(images)

        return StreamingResponse(BytesIO(gif_bytes), media_type="image/gif")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)