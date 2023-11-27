from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import replicate

app = FastAPI()

# Set Replicate API token
replicate.set_auth_token('r8_L0LVdqekP16fCX3Z99ySjrk5wVIkai93nBZju')

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Run the Replicate model with the uploaded image
        output = replicate.run(
            "alaradirik/dreamgaussian:44d1361ed7b4e46754c70de0d91334e79a1bc8bbe3e7ec18835691629de25305",
            input={"image": file.file}
        )
        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
