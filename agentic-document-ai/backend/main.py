from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
from typing import Dict, Any, List
import json

# Import the complete pipeline
from pipeline import pipeline

app = FastAPI(title="Agentic Document AI", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Agentic Document AI API - Production Ready"}

@app.get("/status")
async def get_status():
    """Get pipeline status and capabilities"""
    return pipeline.get_pipeline_status()

@app.post("/uploadDocument")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document through complete 5-layer pipeline"""
    try:
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process through complete pipeline
        result = pipeline.process_document(file_path)
        
        # Add processed image URL to result
        if result.get("status") == "success" and "processed_image_path" in result:
            result["processed_image_url"] = f"http://localhost:8000/processed-image/{os.path.basename(result['processed_image_path'])}"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extractFields")
async def extract_fields(filename: str):
    """Extract fields from processed document (legacy endpoint)"""
    # This endpoint is maintained for compatibility
    # New implementations should use /uploadDocument which includes extraction
    try:
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        result = pipeline.process_document(file_path)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reasonUpdate")
async def reason_update(field_update: Dict[str, Any]):
    """Agent reasoning engine update with real-time behavior"""
    try:
        field_name = field_update.get("field_name")
        new_value = field_update.get("value")
        user_correction = field_update.get("user_correction", False)
        reasoning = field_update.get("reasoning", "")
        
        if not field_name or new_value is None:
            raise HTTPException(status_code=400, detail="field_name and value are required")
        
        # Use pipeline reasoning engine
        result = pipeline.update_field_with_reasoning(
            field_name, new_value, user_correction, reasoning
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/storeCorrection")
async def store_correction(correction_data: Dict[str, Any]):
    """Store user correction for learning memory"""
    try:
        field_name = correction_data.get("field_name")
        original_value = correction_data.get("original_value")
        corrected_value = correction_data.get("corrected_value")
        context = correction_data.get("context", {})
        
        if not all([field_name, original_value, corrected_value]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        result = pipeline.store_correction_for_learning(
            field_name, original_value, corrected_value, context
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getConfidence/{field_name}")
async def get_confidence(field_name: str):
    """Get detailed confidence breakdown for a field"""
    try:
        result = pipeline.get_field_confidence_details(field_name)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processed-image/{filename}")
async def get_processed_image(filename: str):
    """Serve processed image files"""
    file_path = f"processed/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    return FileResponse(file_path, media_type="image/png")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline": pipeline.get_pipeline_status()["status"],
        "version": "1.0.0"
    }

if __name__ == "__main__":
    print("🚀 Starting Agentic Document AI Server...")
    print("📊 5-Layer Pipeline Active")
    print("🧠 Real-time Agent Reasoning Enabled")
    print("🌐 API Available at: http://localhost:8000")
    print("📖 Docs Available at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
