# main.py
import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from data_cleaner import clean_csv
from fastapi.responses import HTMLResponse

app = FastAPI(title="Polars Data Cleaner")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "polars_uploads"
CLEAN_DIR = Path(tempfile.gettempdir()) / "polars_cleaned"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"msg": "Polars Data Cleaner. POST a CSV to /upload to clean and download."}

@app.get("/upload-form", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head><title>Upload CSV</title></head>
        <body>
            <h2>Upload a CSV File for Cleaning</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".csv" required>
                <input type="submit" value="Clean and Download">
            </form>
        </body>
    </html>
    """

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Only allow CSV for now
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # save upload to a temp file
    tmp_in = UPLOAD_DIR / f"{next(tempfile._get_candidate_names())}_{file.filename}"
    with tmp_in.open("wb") as f:
        contents = await file.read()
        f.write(contents)

    # cleaned output path
    out_name = f"cleaned_{file.filename}"
    tmp_out = CLEAN_DIR / out_name

    try:
        # run cleaner (blocking). For very large files you could run this in background/workers
        clean_csv(str(tmp_in), str(tmp_out))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {e}")

    # return cleaned file as attachment
    return FileResponse(str(tmp_out), filename=out_name, media_type="text/csv")

