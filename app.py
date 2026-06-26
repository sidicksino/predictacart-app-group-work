"""
app.py — PredictaMovie Recommendation API
This server loads the movie lists & similarity matrix and exposes a /predict endpoint.
It also serves a built-in HTML test interface at the root URL.
"""
import os
import time
import json
import logging
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List

# ─── Setup Logging ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predictamovie")

# ─── Load the Trained Model Files ───────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", ".")
movies_list_path = os.path.join(MODEL_DIR, "movies_list.pkl")
similarity_path = os.path.join(MODEL_DIR, "similarity.pkl")
logger.info(f"Loading movies list from: {movies_list_path}")
logger.info(f"Loading similarity matrix from: {similarity_path}")

try:
    movies = pickle.load(open(movies_list_path, 'rb'))
    similarity = pickle.load(open(similarity_path, 'rb'))
    logger.info(f"Model loaded! Total movies: {len(movies)}")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

# ─── Create the FastAPI App ────────────────────────────────
app = FastAPI(
    title="PredictaMovie Recommendation API",
    description="Sends a movie title, gets top-5 similar movie recommendations"
)

# ─── CORS Middleware (allows the HTML page to call the API) ─
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Define Request/Response Formats ───────────────────────
class RecommendationRequest(BaseModel):
    movie_title: str = Field(..., min_length=1, example="Iron Man",
        description="Title of the movie to get recommendations for")

# ─── API Endpoints ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the built-in HTML test interface."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/movies")
async def get_movies():
    """Returns a list of all movie titles for frontend autocomplete."""
    try:
        return {"movies": list(movies['title'].values)}
    except Exception as e:
        logger.error(f"Failed to fetch movies list: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check — used by the load balancer to know the server is alive."""
    return {"status": "healthy", "model_loaded": True, "movies_count": len(movies)}

@app.post("/predict")
async def predict(request: RecommendationRequest):
    """
    Send movie title → find similar movies based on cosine similarity tags.
    
    Example request body:
    {
        "movie_title": "The Godfather"
    }
    """
    start_time = time.time()
    
    try:
        # Find index of the movie title (case-insensitive)
        matches = movies[movies['title'].str.lower() == request.movie_title.lower()]
        if matches.empty:
            logger.warning(f"Movie '{request.movie_title}' not found in database.")
            raise HTTPException(status_code=404, detail=f"Movie '{request.movie_title}' not found in database.")
        
        index = matches.index[0]
        logger.info(f"Matching recommendations for movie: {movies.iloc[index].title} (index: {index})")
        
        # Sort similarity scores for this movie in descending order
        # Skip the first element because it is the query movie itself (similarity is 1.0)
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        
        # Get top 5 recommendations
        recommendations = []
        scores = []
        results = []
        for i in distances[1:6]:
            row = movies.iloc[i[0]]
            rec_title = row.title
            rec_id = int(row.id)
            rec_score = round(float(i[1]), 4)
            
            recommendations.append(rec_title)
            scores.append(rec_score)
            results.append({
                "id": rec_id,
                "title": rec_title,
                "score": rec_score
            })
        
        latency = (time.time() - start_time) * 1000
        
        # ─── LOG structured data (visible in CloudWatch Logs) ───
        log_data = {
            "event": "prediction",
            "movie_title": request.movie_title,
            "matched_title": movies.iloc[index].title,
            "latency_ms": round(latency, 2),
            "recommendations": recommendations,
            "scores": scores
        }
        logger.info(json.dumps(log_data))
        
        return {
            "movie_title": movies.iloc[index].title,
            "recommendations": recommendations,
            "scores": scores,
            "results": results,
            "latency_ms": round(latency, 2),
            "model_version": "v1"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Shows details about the loaded recommendation model."""
    return {
        "model_type": "Content-Based Recommender (Cosine Similarity)",
        "total_movies": len(movies),
        "features": ["id", "title", "tags"]
    }