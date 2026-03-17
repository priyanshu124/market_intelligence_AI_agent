# FastAPI Application
# Main entry point for the Market Intelligence AI Agent API

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Market Intelligence AI Agent",
    description="AI-powered market intelligence API"
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Market Intelligence API is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
