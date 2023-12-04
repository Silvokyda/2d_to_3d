from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
import replicate

app = FastAPI()

# Set Replicate API token
replicate = replicate.Client(api_token='r8_dahC2EblDzX9mfM4L0ZYksDMNQZcdfw2n0yi3')

# Define the data folder to store uploaded files temporarily
data_folder = "data"

# Create the data folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Check if the file is not None and has valid content
        if file is None or file.content_type.split("/")[0].lower() != "image":
            return JSONResponse(content={"error": "Invalid or missing image file"}, status_code=400)

        # Save the uploaded file temporarily in the data folder
        file_path = os.path.join(data_folder, file.filename)
        with open(file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Run the Replicate model with the uploaded image
        output = replicate.run(
            "alaradirik/dreamgaussian:44d1361ed7b4e46754c70de0d91334e79a1bc8bbe3e7ec18835691629de25305",
            input={"image": open(file_path, "rb")}
        )

        # Clean up: Remove the temporary file
        os.remove(file_path)

        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
