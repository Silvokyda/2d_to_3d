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

# Function to load models and diffusion configuration
def load_models(models_path):
    xm = load_model('transmitter')
    xm.load_state_dict(torch.load(os.path.join(models_path, 'xm_model.pth')))
    xm.eval()

    text300M = load_model('text300M')
    text300M.load_state_dict(torch.load(os.path.join(models_path, 'text300M_model.pth')))
    text300M.eval()

    diffusion_params = torch.load(os.path.join(models_path, 'diffusion_model.pth'))
    diffusion = GaussianDiffusion(**diffusion_params)

    return xm, text300M, diffusion

# Generate 3D model from the uploaded 2D image
# ...

# Fetch the generated 3D images
@app.get("/get_3d_images")
def get_3d_images():
    # Return the URLs of the generated GIFs
    return {"images": generated_gif_urls}

# ...

# Fetch the generated 3D images
@app.post("/generate_3d_model")
def generate_3d_model_from_image(img: UploadFile = File(...), models_path: str = '/content/saved'):
    try:
        img_path = "/tmp/" + img.filename
        with open(img_path, "wb") as buffer:
            buffer.write(img.file.read())

        xm, text300M, diffusion = load_models(models_path)
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

        render_mode = 'nerf'
        size = 128

        cameras = create_pan_cameras(size, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        images = []
        for i, latent in enumerate(latents):
            frame_images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            images.extend(frame_images)

        gif_bytes = gif_widget(images)

        # Store the URL or data of the generated GIF
        generated_gif_urls.append(gif_bytes)

        return StreamingResponse(BytesIO(gif_bytes), media_type="image/gif")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
