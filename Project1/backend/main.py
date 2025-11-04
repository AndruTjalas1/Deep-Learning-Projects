"""
FastAPI backend bridging your React frontend to Python helpers in MethodFiles.
Only MethodFiles stays the same—import and call it here.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ProcessRequest, ProcessResponse

# ---- import your existing code without changing it ----
# Example: suppose you have MethodFiles/pipeline.py with def run_pipeline(text, **params)
try:
    from MethodFiles.pipeline import run_pipeline  # EDIT to match your actual entry point
except Exception as e:
    run_pipeline = None
    import traceback; traceback.print_exc()

app = FastAPI(title="Project Backend", version="1.0.0")

# CORS is harmless because we’ll also add a Vercel rewrite.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safe because we’ll front it with Vercel rewrites
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "backend"}

@app.post("/api/process", response_model=ProcessResponse)
async def process(data: ProcessRequest):
    if run_pipeline is None:
        raise HTTPException(status_code=500, detail="MethodFiles import failed—check server logs.")
    try:
        # Call your unchanged MethodFiles logic:
        result = run_pipeline(data.input_text or "", **(data.params or {}))
        return ProcessResponse(ok=True, result=result)
    except Exception as e:
        return ProcessResponse(ok=False, result={}, message=str(e))

# Example endpoint for file uploads (if your MethodFiles need files)
@app.post("/api/upload", response_model=ProcessResponse)
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # pass bytes/paths into your MethodFiles as needed:
        # result = run_file_pipeline(contents)
        result = {"filename": file.filename, "size": len(contents)}
        return ProcessResponse(ok=True, result=result)
    except Exception as e:
        return ProcessResponse(ok=False, result={}, message=str(e))
