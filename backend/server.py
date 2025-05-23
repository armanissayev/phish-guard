from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from predict import predict_url

# Initialize FastAPI app
app = FastAPI(
    title="ML API",
    description="API for ML model serving",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class URLRequest(BaseModel):
    url: str

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"status": "healthy", "message": "Server is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Service is operational"}

@app.post("/predict")
async def predict(request: URLRequest):
    """
    Predict whether a URL is legitimate or phishing
    
    This endpoint analyzes the provided URL using three different machine learning models
    and returns the best prediction based on confidence scores.
    """
    try:
        result = predict_url(request.url)
        print(result)
        return result
    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
