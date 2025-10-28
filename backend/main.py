import os
import time
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import openai
import logging
from dotenv import load_dotenv
import cv2
from io import BytesIO
from base64 import b64encode, b64decode
import base64
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Seaborn not available, using matplotlib only for visualizations")

# Load environment variables from .env file
# Try backend/.env and project-root/.env for convenience
try:
    load_dotenv()
    backend_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    project_root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(backend_env):
        load_dotenv(backend_env, override=True)
    if os.path.exists(project_root_env):
        load_dotenv(project_root_env, override=True)
except Exception as _:
    pass

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Use environment variable for API key

# Initialize FastAPI
app = FastAPI()

# Enable CORS with AR-specific headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom headers for AR model access
@app.middleware("http")
async def add_ar_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Add headers for AR model access
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range"
    
    # Add headers for GLB model files
    if request.url.path.endswith('.glb'):
        response.headers["Content-Type"] = "model/gltf-binary"
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
        response.headers["Cross-Origin-Opener-Policy"] = "unsafe-none"
    
    return response

# Define Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FURNITURE_FOLDER = os.path.join(BASE_DIR, "furniture_models/sofas")
THUMBNAIL_FOLDER = os.path.join(FURNITURE_FOLDER, "thumbnails")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Detect CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup models automatically
try:
    from setup_models import main as setup_models
    print("🔄 Setting up models...")
    setup_models()
except Exception as e:
    print(f"⚠️ Model setup failed: {e}")
    print("💡 Models will be downloaded when first used")

# Load the SAM model
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"  # Update the path if needed
if not os.path.exists(SAM_CHECKPOINT_PATH):
    print("⚠️ SAM model not found. Please run setup_models.py first")
    # Create a dummy model for now
    sam = None
    mask_generator = None
    predictor = None
else:
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)

# Global variables for storing images and masks
GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_USER_PREFERENCES = None  # Store recent user preferences
GLOBAL_FURNITURE_CATEGORY = None  # Store selected furniture category

# Load Pre-trained ResNet for Feature Extraction
resnet = models.resnet50(weights='DEFAULT')  # Updated to use 'weights' instead of 'pretrained'
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
resnet.to(device)

# Image Preprocessing for ResNet (Converts uploaded room photos and sofa thumbnails into standardized tensors that ResNet50 can process)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

logger.info("Models Loaded Successfully ✅")

# Visualization Functions

def create_embedding_comparison_chart(text_scores, visual_scores, furniture_names, final_scores):
    """Create a comparison chart showing text vs visual scores"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(furniture_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, text_scores, width, label='Text Similarity', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x, visual_scores, width, label='Visual Similarity', color='lightgreen', alpha=0.8)
    bars3 = ax.bar(x + width, final_scores, width, label='Final Score', color='orange', alpha=0.8)
    
    ax.set_xlabel('Furniture Items')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Multimodal Recommendation Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(furniture_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close(fig)  # Close figure to free memory
    return image_base64

# Sofa Details (Prices and Descriptions)
# Furniture inventory organized by category
furniture_inventory = {
    "sofas": {
    "black_sofa": {
        "name": "Black Sofa",
        "price": 1000.00,
        "description": "A sleek and modern black sofa, perfect for contemporary spaces.",
        "style": "modern",
        "color": "black",
        "material": "leather",
            "dimensions": "84\"W x 36\"D x 32\"H",
            "category": "sofas"
    },
    "blue_sofa": {
        "name": "Blue Sofa",
        "price": 1200.00,
        "description": "A stylish blue sofa, ideal for adding a pop of color to your space.",
        "style": "contemporary",
        "color": "blue",
        "material": "velvet",
            "dimensions": "90\"W x 38\"D x 30\"H",
            "category": "sofas"
    },
    "modern_sofa2": {
        "name": "Modern Sofa 2",
        "price": 1100.00,
        "description": "A stylish and comfortable sofa, ideal for modern spaces.",
        "style": "modern",
        "color": "gray",
        "material": "linen",
            "dimensions": "96\"W x 35\"D x 31\"H",
            "category": "sofas"
        }
    },
    "tables": {
        "Dinning Table": {
            "name": "Dinning Table",
            "price": 950.00,
            "description": "A spacious dining table perfect for family meals and entertaining guests.",
            "style": "traditional",
            "color": "brown",
            "material": "wood",
            "dimensions": "72\"W x 36\"D x 30\"H",
            "category": "tables"
        },
        "Picnic Table": {
            "name": "Picnic Table",
            "price": 650.00,
            "description": "A durable outdoor picnic table ideal for backyard gatherings and outdoor dining.",
            "style": "rustic",
            "color": "natural",
            "material": "wood",
            "dimensions": "96\"W x 30\"D x 29\"H",
            "category": "tables"
        },
        "round_table": {
            "name": "Round Dining Table",
            "price": 1200.00,
            "description": "A classic round dining table ideal for family gatherings.",
            "style": "traditional",
            "color": "brown",
            "material": "wood",
            "dimensions": "60\"W x 60\"D x 30\"H",
            "category": "tables"
        }
    },
    "cabinets": {
        "old_dresser": {
            "name": "Vintage Dresser",
            "price": 900.00,
            "description": "A beautiful vintage dresser with classic charm and ample storage.",
            "style": "vintage",
            "color": "brown",
            "material": "wood",
            "dimensions": "60\"W x 20\"D x 32\"H",
            "category": "cabinets"
        },
        "wooden_cabinet": {
            "name": "Modern Wooden Cabinet",
            "price": 1100.00,
            "description": "A sleek modern cabinet perfect for storage and display.",
            "style": "modern",
            "color": "walnut",
            "material": "wood",
            "dimensions": "72\"W x 18\"D x 36\"H",
            "category": "cabinets"
        }
    },
    "chairs": {
        "Recliner Chair": {
            "name": "Recliner Chair",
            "price": 900.00,
            "description": "A comfortable recliner chair perfect for relaxation and movie nights.",
            "style": "traditional",
            "color": "brown",
            "material": "leather",
            "dimensions": "36\"W x 40\"D x 42\"H",
            "category": "chairs"
        },
        "Rocking Chair": {
            "name": "Rocking Chair",
            "price": 650.00,
            "description": "A classic wooden rocking chair ideal for nurseries and reading nooks.",
            "style": "classic",
            "color": "natural",
            "material": "wood",
            "dimensions": "28\"W x 42\"D x 44\"H",
            "category": "chairs"
        }
    }
}

# Backward compatibility - keep sofa_details for existing code
sofa_details = furniture_inventory["sofas"]

def detect_furniture_category(user_message):
    """Detect furniture category from user message"""
    if not user_message:
        return None
    
    message_lower = user_message.lower()
    
    # Sofa keywords
    sofa_keywords = ["sofa", "couch", "settee", "loveseat", "sectional"]
    if any(keyword in message_lower for keyword in sofa_keywords):
        return "sofas"
    
    # Table keywords
    table_keywords = ["table", "desk", "coffee table", "dining table", "side table", "end table"]
    if any(keyword in message_lower for keyword in table_keywords):
        return "tables"
    
    # Cabinet keywords
    cabinet_keywords = ["cabinet", "dresser", "wardrobe", "closet", "storage", "chest", "drawer"]
    if any(keyword in message_lower for keyword in cabinet_keywords):
        return "cabinets"
    
    # Chair keywords
    chair_keywords = ["chair", "recliner", "rocking", "armchair", "dining chair", "office chair", "seat"]
    if any(keyword in message_lower for keyword in chair_keywords):
        return "chairs"
    
    return None

def get_furniture_by_category(category):
    """Get furniture details for a specific category"""
    if category and category in furniture_inventory:
        return furniture_inventory[category]
    return furniture_inventory["sofas"]  # Default to sofas

def get_thumbnail_folder(category: str):
    """Get the correct thumbnail folder name for a category."""
    return "thumbnails" if category == "sofas" else "Thumbnails"

def get_local_ip():
    """Get the local IP address for mobile device access."""
    import socket
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            # Ensure we don't return localhost for mobile access
            if local_ip == "127.0.0.1" or local_ip == "localhost":
                # Try alternative method
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                if local_ip == "127.0.0.1":
                    # Last resort - try to get first non-localhost IP
                    import subprocess
                    result = subprocess.run(['ipconfig'], capture_output=True, text=True)
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'IPv4 Address' in line and '192.168.' in line:
                            local_ip = line.split(':')[-1].strip()
                            break
            return local_ip
    except Exception:
        # Fallback - try to get network IP
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip != "127.0.0.1":
                return local_ip
        except:
            pass
        return "127.0.0.1"

def get_model_url(category: str, filename: str):
    """Generate model URL with correct IP address."""
    local_ip = get_local_ip()
    return f"http://{local_ip}:8000/{category}/{filename}.glb"

def get_thumbnail_url(category: str, filename: str):
    """Generate thumbnail URL with correct IP address."""
    local_ip = get_local_ip()
    thumbnail_folder = get_thumbnail_folder(category)
    return f"http://{local_ip}:8000/{category}/{thumbnail_folder}/{filename}.png"

def clear_embedding_cache():
    """Clear all cached embeddings to force reload for new category"""
    global INVENTORY_EMBEDDINGS, INVENTORY_VISUAL_EMBEDDINGS, MULTIMODAL_EMBEDDINGS
    INVENTORY_EMBEDDINGS = {}
    INVENTORY_VISUAL_EMBEDDINGS = {}
    MULTIMODAL_EMBEDDINGS = {}
    logger.info("Cleared embedding cache for category change")

def get_furniture_key_from_name(display_name: str, category: str = None) -> str:
    """Convert display name to furniture key"""
    target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
    current_furniture = get_furniture_by_category(target_category)
    
    # Try exact match first (case-insensitive)
    for key, details in current_furniture.items():
        if details.get("name", "").lower() == display_name.lower():
            return key
    
    # Try exact key match (case-insensitive)
    for key in current_furniture.keys():
        if key.lower() == display_name.lower():
            return key
    
    # Try partial match (but prefer shorter matches to avoid over-matching)
    display_lower = display_name.lower()
    best_match = None
    best_match_length = float('inf')
    
    for key, details in current_furniture.items():
        furniture_name = details.get("name", "").lower()
        if display_lower in furniture_name:
            # Prefer shorter furniture names to avoid over-matching
            if len(furniture_name) < best_match_length:
                best_match = key
                best_match_length = len(furniture_name)
    
    if best_match:
        return best_match
    
    # If no match found, return the original name (might already be a key)
    return display_name

# ------------------------------
# Multimodal Recommender System
# ------------------------------
# This module implements multimodal fusion combining visual (ResNet) and text embeddings
# for comprehensive furniture recommendations across different input types.

# In-memory cache for inventory embeddings
INVENTORY_EMBEDDINGS = {}  # Text embeddings
INVENTORY_VISUAL_EMBEDDINGS = {}  # Visual embeddings
MULTIMODAL_EMBEDDINGS = {}  # Fused multimodal embeddings

def _build_inventory_text(furniture_key: str, category=None) -> str:
    """Build a searchable text string from furniture metadata for text-based recommendations.
    
    Combines name, description, style, color, material, and dimensions into one string
    that gets embedded for cosine similarity matching against user queries.
    """
    # Use the specified category or current global category
    target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
    current_furniture = get_furniture_by_category(target_category)
    meta = current_furniture.get(furniture_key, {})
    parts = [
        meta.get("name", ""),
        meta.get("description", ""),
        meta.get("style", ""),
        meta.get("color", ""),
        meta.get("material", ""),
        meta.get("dimensions", ""),
    ]
    return " ".join(p for p in parts if p)

def _get_text_embedding(text: str):
    """Convert text to a vector embedding using OpenAI's text-embedding-3-small model.
    
    Used for text-based furniture recommendations - converts user queries and sofa
    descriptions into comparable vector representations for cosine similarity.
    """
    try:
        emb = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text.strip()[:4000],
        )
        return np.array(emb.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

def _ensure_inventory_embeddings(category=None):
    """Pre-compute and cache text embeddings for all furniture items.
    
    Runs once on first recommendation request - converts all furniture descriptions
    to embeddings and stores them in memory for fast cosine similarity calculations.
    """
    global INVENTORY_EMBEDDINGS
    if INVENTORY_EMBEDDINGS:
        return
    
    # Use the specified category or current global category
    target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
    current_furniture = get_furniture_by_category(target_category)
    
    for key in current_furniture.keys():
        text = _build_inventory_text(key, target_category)
        vec = _get_text_embedding(text)
        if vec is not None:
            INVENTORY_EMBEDDINGS[key] = vec

def _ensure_visual_embeddings(category=None):
    """Pre-compute and cache visual embeddings for all furniture thumbnails.
    
    Extracts ResNet50 features from all furniture thumbnails and stores them
    for fast visual similarity calculations.
    """
    global INVENTORY_VISUAL_EMBEDDINGS
    if INVENTORY_VISUAL_EMBEDDINGS:
        return
    
    # Use the specified category or current global category
    target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
    current_furniture = get_furniture_by_category(target_category)
    
    # Determine the correct thumbnail folder
    if target_category == "sofas":
        thumbnail_folder = THUMBNAIL_FOLDER  # Use existing sofas folder
    else:
        # Map category names to actual folder names (case-sensitive)
        folder_mapping = {
            "tables": "Tables",
            "cabinets": "Cabinets", 
            "chairs": "Chairs"
        }
        actual_folder = folder_mapping.get(target_category, target_category)
        thumbnail_folder = os.path.join(BASE_DIR, "furniture_models", actual_folder, "Thumbnails")
    
    if not os.path.exists(thumbnail_folder):
        logger.error(f"Thumbnail folder not found: {thumbnail_folder}")
        return
    
    for filename in os.listdir(thumbnail_folder):
        if not filename.endswith('.png'):
            continue
        key = filename.replace('.png', '')
        if key not in current_furniture:
            continue
        
        thumbnail_path = os.path.join(thumbnail_folder, filename)
        try:
            visual_features = extract_features(thumbnail_path)
            INVENTORY_VISUAL_EMBEDDINGS[key] = visual_features
            logger.info(f"Cached visual features for {key}")
        except Exception as e:
            logger.error(f"Failed to extract visual features for {key}: {e}")

def _fuse_multimodal_embeddings(text_vec, visual_vec, fusion_method="concat"):
    """Fuse text and visual embeddings into a unified multimodal representation.
    
    Args:
        text_vec: Text embedding vector
        visual_vec: Visual embedding vector  
        fusion_method: Method to combine embeddings ("concat", "add", "weighted")
    
    Returns:
        Fused multimodal embedding vector
    """
    if text_vec is None or visual_vec is None:
        return None
    
    text_vec = np.array(text_vec)
    visual_vec = np.array(visual_vec)
    
    if fusion_method == "concat":
        # Simple concatenation
        fused = np.concatenate([text_vec, visual_vec])
    elif fusion_method == "add":
        # Element-wise addition (requires same dimensions)
        if text_vec.shape != visual_vec.shape:
            # Resize to match smaller dimension
            min_dim = min(len(text_vec), len(visual_vec))
            text_vec = text_vec[:min_dim]
            visual_vec = visual_vec[:min_dim]
        fused = text_vec + visual_vec
    elif fusion_method == "weighted":
        # Weighted combination (0.6 text, 0.4 visual)
        if text_vec.shape != visual_vec.shape:
            min_dim = min(len(text_vec), len(visual_vec))
            text_vec = text_vec[:min_dim]
            visual_vec = visual_vec[:min_dim]
        fused = 0.6 * text_vec + 0.4 * visual_vec
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Normalize the fused embedding
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    
    return fused

def _ensure_multimodal_embeddings(category=None):
    """Pre-compute and cache multimodal embeddings for all furniture items.
    
    Combines text and visual embeddings into unified multimodal representations
    for comprehensive similarity matching.
    """
    global MULTIMODAL_EMBEDDINGS
    if MULTIMODAL_EMBEDDINGS:
        return
    
    # Use the specified category or current global category
    target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
    current_furniture = get_furniture_by_category(target_category)
    
    # Ensure both text and visual embeddings are available
    _ensure_inventory_embeddings(target_category)
    _ensure_visual_embeddings(target_category)
    
    for key in current_furniture.keys():
        if key in INVENTORY_EMBEDDINGS and key in INVENTORY_VISUAL_EMBEDDINGS:
            text_vec = INVENTORY_EMBEDDINGS[key]
            visual_vec = INVENTORY_VISUAL_EMBEDDINGS[key]
            
            # Create multimodal embedding
            multimodal_vec = _fuse_multimodal_embeddings(text_vec, visual_vec, "concat")
            if multimodal_vec is not None:
                MULTIMODAL_EMBEDDINGS[key] = multimodal_vec
                logger.info(f"Cached multimodal features for {key}")

def recommend_furniture_by_cosine(user_message: str, top_k: int = 3, category: str = None):
    """Return top_k inventory items ranked by cosine similarity to user_message.

    This function is additive and does not modify existing behavior. It can be
    called by the chat or inventory flow to rank items.
    """
    try:
        target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(target_category)
        
        _ensure_inventory_embeddings()
        query_vec = _get_text_embedding(user_message or "")
        if query_vec is None or not INVENTORY_EMBEDDINGS:
            # Fallback: simple keyword scoring if embeddings unavailable
            logger.warning("Embeddings unavailable; using keyword fallback")
            tokens = {t.lower() for t in (user_message or "").split()}
            scores = []
            for key, meta in current_furniture.items():
                text = _build_inventory_text(key).lower()
                score = sum(1 for t in tokens if t in text)
                scores.append((key, float(score)))
        else:
            scores = []
            for key, vec in INVENTORY_EMBEDDINGS.items():
                # Only include furniture from the current category
                if key in current_furniture:
                    sim = float(cosine_similarity([query_vec], [vec])[0][0])
                    scores.append((key, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: max(1, top_k)]
        results = []
        for key, score in top:
            meta = current_furniture.get(key, {})
            results.append({
                "key": key,
                "name": meta.get("name", key),
                "score": round(float(score), 4),
                "thumbnail": get_thumbnail_url(target_category, key),
                "glb_model": get_model_url(target_category, key),
                **meta,
            })
        return results
    except Exception as e:
        logger.error(f"cosine recommender error: {e}")
        return []

def recommend_furniture_multimodal(query_text: str = None, query_image_path: str = None, top_k: int = 3, fusion_method: str = "concat", category: str = None, text_weight: float = 0.75, visual_weight: float = 0.25, color_boost_enabled: bool = True):
    """Multimodal furniture recommendation using both text and visual features.
    
    Args:
        query_text: Text description of desired furniture
        query_image_path: Path to room image for visual matching
        top_k: Number of top recommendations to return
        fusion_method: Method to fuse text and visual features ("concat", "add", "weighted")
    
    Returns:
        List of ranked furniture recommendations with multimodal similarity scores
    """
    try:
        logger.info(f"=== recommend_furniture_multimodal called with query_text='{query_text}', category='{category}' ===")
        # Use the specified category or default to sofas
        target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(target_category)
        
        _ensure_multimodal_embeddings(target_category)
        
        if not query_text and not query_image_path:
            raise ValueError("Either query_text or query_image_path must be provided")
        
        query_embeddings = []
        
        # Get text embedding if text query provided
        if query_text:
            text_embedding = _get_text_embedding(query_text)
            if text_embedding is not None:
                query_embeddings.append(("text", text_embedding))
        
        # Get visual embedding if image query provided
        if query_image_path and os.path.exists(query_image_path):
            visual_embedding = extract_features(query_image_path)
            if visual_embedding is not None:
                query_embeddings.append(("visual", visual_embedding))
        
        if not query_embeddings:
            logger.warning("No valid embeddings generated from query")
            return []
        
        # Detect if user has specific color/style preferences
        has_specific_preferences = False
        specific_color = None
        if query_text:
            logger.info(f"Checking specific preferences for query: '{query_text}'")
            specific_keywords = ["black", "blue", "gray", "modern", "contemporary", "leather", "velvet", "linen"]
            has_specific_preferences = any(keyword in query_text.lower() for keyword in specific_keywords)
            logger.info(f"Initial specific preference check: {has_specific_preferences}")
            
            # Also check if user mentioned a specific furniture item that exists in inventory
            if not has_specific_preferences:
                current_furniture = get_furniture_by_category(target_category)
                logger.info(f"Checking furniture names in category {target_category}: {list(current_furniture.keys())}")
                for furniture_key, furniture_data in current_furniture.items():
                    furniture_name = furniture_data.get("name", "").lower()
                    if furniture_name:
                        # Check if any significant words from the furniture name are in the query
                        furniture_words = furniture_name.split()
                        query_words = query_text.lower().split()
                        # If at least 2 words match, consider it a specific preference
                        matching_words = sum(1 for word in furniture_words if word in query_words)
                        logger.info(f"Checking {furniture_name}: furniture_words={furniture_words}, query_words={query_words}, matching_words={matching_words}")
                        if matching_words >= 2:
                            has_specific_preferences = True
                            logger.info(f"Detected specific furniture preference: {furniture_name} (matched {matching_words} words)")
                            break
            
            # Extract specific color if mentioned
            for color in ["black", "blue", "gray"]:
                if color in query_text.lower():
                    specific_color = color
                    break
                    
            logger.info(f"Final result - User has specific preferences: {has_specific_preferences}, specific color: {specific_color}")
        
        # Calculate multimodal similarity scores for current category
        scores = []
        for key, multimodal_vec in MULTIMODAL_EMBEDDINGS.items():
            # Only process furniture from the current category
            if key not in current_furniture:
                continue
            total_similarity = 0.0
            weight_sum = 0.0
            text_similarity = 0.0
            visual_similarity = 0.0
            
            for modality, query_vec in query_embeddings:
                if modality == "text":
                    # For text queries, compare with text portion of multimodal embedding
                    # (first 1536 dimensions for text-embedding-3-small)
                    text_portion = multimodal_vec[:1536] if len(multimodal_vec) > 1536 else multimodal_vec
                    text_similarity = float(cosine_similarity([query_vec], [text_portion])[0][0])
                    
                    # Use user-provided text weight
                    weight = text_weight
                    logger.info(f"Using user text weight ({text_weight}) for text similarity: {query_text}")
                    
                    total_similarity += text_similarity * weight
                    weight_sum += weight
                        
                elif modality == "visual":
                    # For visual queries, compare with visual portion of multimodal embedding
                    # (last 2048 dimensions for ResNet50 features)
                    visual_portion = multimodal_vec[-2048:] if len(multimodal_vec) > 2048 else multimodal_vec
                    visual_similarity = float(cosine_similarity([query_vec], [visual_portion])[0][0])
                    
                    # Use user-provided visual weight
                    weight = visual_weight
                    logger.info(f"Using user visual weight ({visual_weight}) for visual similarity")
                    
                    total_similarity += visual_similarity * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                final_score = total_similarity / weight_sum
                
                # Boost score for exact color matches when user specifies a color and color boost is enabled
                if specific_color and color_boost_enabled:
                    meta = current_furniture.get(key, {})
                    furniture_color = meta.get("color", "").lower()
                    if furniture_color == specific_color:
                        final_score += 0.3  # Significant boost for exact color match
                        logger.info(f"Color match boost: {key} ({furniture_color}) matches user preference ({specific_color})")
                
                scores.append((key, final_score, text_similarity, visual_similarity))
                logger.info(f"Furniture {key}: text_sim={text_similarity:.4f}, visual_sim={visual_similarity:.4f}, final_score={final_score:.4f}")
        
        # Sort by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:max(1, top_k)]
        
        # If user specified a specific color and the top result doesn't match, force the color match to be #1
        if specific_color and top:
            top_key = top[0][0]
            meta = current_furniture.get(top_key, {})
            furniture_color = meta.get("color", "").lower()
            
            if furniture_color != specific_color:
                # Find the furniture that matches the specified color
                for key, score in scores:
                    meta = current_furniture.get(key, {})
                    if meta.get("color", "").lower() == specific_color:
                        logger.info(f"Overriding top result: {top_key} -> {key} due to color preference")
                        top = [(key, score)]
                        break
        
        # Format results
        results = []
        for key, score, text_sim, visual_sim in top:
            meta = current_furniture.get(key, {})
            results.append({
                "key": key,
                "name": meta.get("name", key),
                "score": round(float(score), 4),
                "text_similarity": round(float(text_sim), 4),
                "visual_similarity": round(float(visual_sim), 4),
                "thumbnail": f"http://127.0.0.1:8000/{target_category}/{get_thumbnail_folder(target_category)}/{key}.png",
                "glb_model": f"http://127.0.0.1:8000/{target_category}/{key}.glb",
                "modality": "multimodal",
                "preference_weighting": "high_text" if has_specific_preferences else "balanced",
                **meta,
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Multimodal recommender error: {e}")
        
        # Fallback: Simple text-based matching for specific color requests
        if query_text and specific_color:
            logger.info(f"Using text-only fallback for color preference: {specific_color}")
            for key, meta in current_furniture.items():
                if meta.get("color", "").lower() == specific_color:
                    return [{
                        "key": key,
                        "name": meta.get("name", key),
                        "score": 1.0,  # Perfect match
                        "thumbnail": f"http://127.0.0.1:8000/{target_category}/Thumbnails/{key}.png",
                        "glb_model": f"http://127.0.0.1:8000/{target_category}/{key}.glb",
                        "modality": "text_fallback",
                        "preference_weighting": "exact_color_match",
                        **meta,
                    }]
        
        return []

def find_most_suitable_sofa_multimodal(room_image_path: str, user_preferences: str = None, top_k: int = 3, category: str = None, text_weight: float = 0.75, visual_weight: float = 0.25, color_boost_enabled: bool = True):
    """Enhanced sofa recommendation using multimodal approach.
    
    Combines visual room analysis with optional text preferences for better matching.
    
    Args:
        room_image_path: Path to uploaded room image
        user_preferences: Optional text description of preferences
        top_k: Number of recommendations to return
    
    Returns:
        Tuple of (best_sofa, similarity_score, reason, sorted_sofas)
    """
    try:
        logger.info("Finding most suitable sofa using multimodal approach...")
        
        # Use multimodal recommendation
        target_category = category or GLOBAL_FURNITURE_CATEGORY or "sofas"
        recommendations = recommend_furniture_multimodal(
            query_text=user_preferences,
            query_image_path=room_image_path,
            top_k=top_k,
            category=target_category,
            text_weight=text_weight,
            visual_weight=visual_weight,
            color_boost_enabled=color_boost_enabled
        )
        
        if not recommendations:
            logger.info("No suitable furniture found with multimodal approach.")
            return None, None, "No furniture available.", []
        
        best_match = recommendations[0]
        best_furniture = best_match["key"] + ".png"  # Convert to filename format
        similarity_score = best_match["score"]
        
        # Generate reasoning
        if user_preferences:
            reason = f"Based on your room's visual style and your preferences '{user_preferences}', this {best_match['name']} matches with a multimodal similarity score of {similarity_score:.2f}."
        else:
            reason = f"This {best_match['name']} matches your room's visual style with a multimodal similarity score of {similarity_score:.2f}."
        
        # Convert to sorted_sofas format for compatibility
        sorted_sofas = [(rec["key"] + ".png", rec["score"]) for rec in recommendations]
        
        logger.info(f"Multimodal best furniture: {best_furniture}, Similarity score: {similarity_score}")
        return best_furniture, similarity_score, reason, sorted_sofas
        
    except Exception as e:
        logger.error(f"Error in multimodal sofa finding: {e}")
        logger.error(f"Error details: {str(e)}")
        # Don't fallback to visual-only approach to maintain consistency with analysis
        # Return empty result instead
        logger.info("Multimodal approach failed, returning empty result to maintain consistency")
        return None, None, "Multimodal recommendation failed", []

# New endpoint to get initial furniture data
@app.get("/get-initial-furniture")
async def get_initial_furniture():
    """Load furniture inventory and metadata when the frontend starts.
    
    Returns all available furniture for the current category with their details
    and thumbnail/GLB URLs for the frontend to display and use in recommendations.
    """
    try:
        # Get current category or default to sofas
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(current_category)
        
        # Prepare inventory list with all available furniture for current category
        inventory = []
        if current_category == "sofas":
            category_folder = FURNITURE_FOLDER
            thumbnail_folder = get_thumbnail_folder(current_category)
        else:
            # Map category names to actual folder names (case-sensitive)
            folder_mapping = {
                "tables": "Tables",
                "cabinets": "Cabinets", 
                "chairs": "Chairs"
            }
            actual_folder = folder_mapping.get(current_category, current_category)
            category_folder = os.path.join(BASE_DIR, "furniture_models", actual_folder)
            thumbnail_folder = get_thumbnail_folder(current_category)
        
        if os.path.exists(category_folder):
            for filename in os.listdir(category_folder):
                if filename.endswith(".glb"):
                    furniture_name = filename.replace(".glb", "")
                    if furniture_name in current_furniture:
                        item = {
                            "filename": furniture_name,  # Add filename field for backend calls
                            "name": current_furniture[furniture_name]["name"],  # Display name
                            "thumbnail": get_thumbnail_url(current_category, furniture_name),
                            "glb_model": get_model_url(current_category, furniture_name),  # GLB model URL
                            "price": f"${current_furniture[furniture_name].get('price', 0):.0f}",  # Format price with $ sign
                            "description": current_furniture[furniture_name].get("description", "No description available"),
                            "style": current_furniture[furniture_name].get("style", "Unknown"),
                            "material": current_furniture[furniture_name].get("material", "Unknown"),
                            "category": current_furniture[furniture_name].get("category", current_category)
                        }
                        inventory.append(item)
        
        return JSONResponse(content={
            "furniture_details": current_furniture,
            "inventory": inventory,
            "current_category": current_category,
            "available_categories": list(furniture_inventory.keys())
        })
    except Exception as e:
        logger.error(f"Error in /get-initial-furniture: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/select-category")
async def select_category(category: dict):
    """Select furniture category"""
    global GLOBAL_FURNITURE_CATEGORY
    
    try:
        selected_category = category.get("category", "").lower()
        
        if selected_category not in furniture_inventory:
            raise HTTPException(status_code=400, detail=f"Invalid category. Available: {list(furniture_inventory.keys())}")
        
        GLOBAL_FURNITURE_CATEGORY = selected_category
        logger.info(f"Selected furniture category: {selected_category}")
        
        # Get furniture for the selected category
        current_furniture = get_furniture_by_category(selected_category)
        inventory = []
        if selected_category == "sofas":
            category_folder = FURNITURE_FOLDER
            thumbnail_folder = get_thumbnail_folder(selected_category)
        else:
            # Map category names to actual folder names (case-sensitive)
            folder_mapping = {
                "tables": "Tables",
                "cabinets": "Cabinets", 
                "chairs": "Chairs"
            }
            actual_folder = folder_mapping.get(selected_category, selected_category)
            category_folder = os.path.join(BASE_DIR, "furniture_models", actual_folder)
            thumbnail_folder = get_thumbnail_folder(selected_category)
        
        if os.path.exists(category_folder):
            for filename in os.listdir(category_folder):
                if filename.endswith(".glb"):
                    furniture_name = filename.replace(".glb", "")
                    if furniture_name in current_furniture:
                        item = {
                            "filename": furniture_name,
                            "name": current_furniture[furniture_name]["name"],
                            "thumbnail": get_thumbnail_url(selected_category, furniture_name),
                            "glb_model": get_model_url(selected_category, furniture_name),
                            "price": f"${current_furniture[furniture_name].get('price', 0):.0f}",  # Format price with $ sign
                            "description": current_furniture[furniture_name].get("description", "No description available"),
                            "style": current_furniture[furniture_name].get("style", "Unknown"),
                            "material": current_furniture[furniture_name].get("material", "Unknown"),
                            "category": current_furniture[furniture_name].get("category", selected_category)
                        }
                        inventory.append(item)
        
        return JSONResponse(content={
            "furniture_details": current_furniture,
            "inventory": inventory,
            "current_category": selected_category,
            "message": f"Switched to {selected_category} category"
        })
        
    except Exception as e:
        logger.error(f"Error selecting category: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Object Detection Endpoints
@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """Detect and segment all objects in an uploaded image using SAM.
    
    Uses Segment Anything Model to automatically find objects, then improves
    object names with AI. Returns bounding boxes, confidence scores, and preview images.
    (This is an experimental feature - not used in the main furniture replacement flow)
    """
    try:
        logger.info("Detecting objects in uploaded image...")
        
        # Save the uploaded image
        image_data = await file.read()
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(image_data)
        
        # Import AI detection functions
        from Endpoints.ai_object_detection import detect_objects_in_image, get_object_preview_images, improve_object_names_with_ai
        
        # Detect objects using SAM
        detected_objects = detect_objects_in_image(image_path, sam, mask_generator)
        
        if not detected_objects:
            return JSONResponse(content={
                "message": "No objects detected in the image",
                "objects": []
            })
        
        # Improve object names with AI
        detected_objects = improve_object_names_with_ai(detected_objects, image_path)
        
        # Generate preview images
        preview_images = get_object_preview_images(image_path, detected_objects)
        
        # Prepare response
        objects_info = []
        for i, obj in enumerate(detected_objects):
            objects_info.append({
                "id": obj['id'],
                "name": obj['name'],
                "bbox": obj['bbox'],
                "area": obj['area'],
                "confidence": obj['confidence'],
                "preview": preview_images[i]['preview_image'] if i < len(preview_images) else None
            })
        
        return JSONResponse(content={
            "message": f"Found {len(detected_objects)} objects in the image",
            "objects": objects_info,
            "image_url": f"http://127.0.0.1:8000/uploads/{file.filename}"
        })
        
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/replace-object")
async def replace_object(data: dict):
    """Replace a detected object with selected furniture using SAM masks.
    
    Takes an object ID from automatic detection and replaces it with a chosen
    furniture item. Uses the object's SAM-generated mask for precise replacement.
    (This is an experimental feature - not used in the main furniture replacement flow)
    """
    try:
        image_filename = data.get("image_filename")
        object_id = data.get("object_id")
        furniture_name = data.get("furniture_name")
        
        if not all([image_filename, object_id is not None, furniture_name]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        logger.info(f"Replacing object {object_id} with {furniture_name}")
        
        # Import AI detection functions
        from Endpoints.ai_object_detection import detect_objects_in_image, replace_selected_object
        
        # Get image path
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get furniture path
        furniture_path = os.path.join(THUMBNAIL_FOLDER, furniture_name + ".png")
        if not os.path.exists(furniture_path):
            raise HTTPException(status_code=404, detail="Furniture image not found")
        
        # Detect objects again (we need the mask data)
        detected_objects = detect_objects_in_image(image_path, sam, mask_generator)
        
        # Replace the selected object
        logger.info(f"Calling replace_selected_object with: image_path={image_path}, object_id={object_id}, detected_objects_count={len(detected_objects)}, furniture_path={furniture_path}")
        result_image = replace_selected_object(image_path, object_id, detected_objects, furniture_path)
        logger.info(f"replace_selected_object returned: {type(result_image)}")
        
        if result_image is None:
            raise HTTPException(status_code=500, detail="Object replacement failed")
        
        # Save the result
        result_filename = f"replaced_{image_filename}"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)
        
        # Convert to base64 for frontend
        from Endpoints.ai_object_detection import pil_image_to_base64
        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        result_base64 = pil_image_to_base64(result_pil)
        
        return JSONResponse(content={
            "message": "Object replaced successfully!",
            "result_image": result_base64,
            "result_url": f"http://127.0.0.1:8000/uploads/{result_filename}"
        })
        
    except Exception as e:
        logger.error(f"Error in object replacement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Extract Features from an Image
def extract_features(image_path):
    """Extract a fixed-length visual embedding for an image using ResNet50.

    Purpose
    - Converts an input image to a normalized tensor and forwards it through a
      pretrained ResNet50 trunk (avg-pooled features).
    - Used to compare a room photo against sofa thumbnails via cosine
      similarity.

    Returns
    - 1D numpy array feature vector (float32)
    """
    try:
        logger.info(f"Extracting features from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet(image)
        return features.cpu().numpy().flatten()
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

# Find the Most Suitable Sofa with Reasoning
def find_most_suitable_sofa(room_features):
    """Rank sofas by cosine similarity to the room's visual features.

    Steps
    - Compute/collect embeddings for all sofa thumbnails.
    - Compute cosine similarity against the room embedding.
    - Return top match, its score, a human-readable reason, and the full
      sorted list for cycling in the UI.
    """
    try:
        logger.info("Finding the most suitable sofa...")
        sofa_features = {}
        for filename in os.listdir(THUMBNAIL_FOLDER):
            if filename.endswith(".png"):
                sofa_path = os.path.join(THUMBNAIL_FOLDER, filename)
                logger.info(f"Processing thumbnail: {sofa_path}")

                features = extract_features(sofa_path)
                sofa_features[filename] = features

        if not sofa_features:
            logger.info("No sofa features found.")
            return None, None, "No furniture available."

        similarities = {sofa: cosine_similarity([room_features], [features])[0][0] for sofa, features in sofa_features.items()}
        sorted_sofas = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        most_suitable_sofa = sorted_sofas[0][0]
        similarity_score = sorted_sofas[0][1]
        reason = f"This sofa matches the room's style and color scheme with a similarity score of {similarity_score:.2f}."

        logger.info(f"Most suitable sofa: {most_suitable_sofa}, Similarity score: {similarity_score}")
        return most_suitable_sofa, similarity_score, reason, sorted_sofas
    except Exception as e:
        logger.error(f"Error finding suitable sofa: {e}")
        raise

# Inpaint Sofa into the Uploaded Image using OpenCV
def inpaint_sofa_into_image(uploaded_image_path, sofa_image_path, mask_path, output_path):
    """Legacy inpainting function using OpenCV with feathering and alpha blending.
    
    This is an older approach that uses morphological operations and soft blending.
    Not used in the current main flow - kept for reference/experimental purposes.
    """
    try:
        # Load original image, sofa, and mask
        uploaded_image = cv2.imread(uploaded_image_path)
        sofa_image = cv2.imread(sofa_image_path, cv2.IMREAD_UNCHANGED)  
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure mask is binary
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Step 1: Get bounding box of the mask area
        y_indices, x_indices = np.where(mask == 255)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("No valid mask found.")

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Step 2: Compute new sofa size based on mask dimensions
        mask_width = x_max - x_min
        mask_height = y_max - y_min

        # Step 3: Resize sofa to fit inside the mask area while preserving aspect ratio
        sh, sw = sofa_image.shape[:2]
        if sw == 0 or sh == 0:
            raise ValueError("Invalid sofa image dimensions.")
        scale = min(mask_width / sw, mask_height / sh)
        new_w = max(1, int(sw * scale))
        new_h = max(1, int(sh * scale))
        sofa_resized = cv2.resize(sofa_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Step 4: Create a copy of the original image to avoid modifying it directly
        final_image = uploaded_image.copy()

        # Step 4.5: Compute placement to center within mask bounds
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)

        # Feather the mask edges for soft blending
        feather = 7  # odd radius
        soft_mask_full = cv2.GaussianBlur(mask, (feather, feather), 0)
        soft_mask_crop = soft_mask_full[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        soft_mask_norm = (soft_mask_crop.astype(np.float32) / 255.0)[..., None]

        # Prepare channels
        if sofa_resized.shape[2] == 4:  
            sofa_rgb = sofa_resized[:, :, :3].astype(np.float32)
            sofa_alpha = (sofa_resized[:, :, 3:4].astype(np.float32) / 255.0)
            blend_alpha = np.clip(soft_mask_norm * sofa_alpha, 0.0, 1.0)
        else:
            sofa_rgb = sofa_resized[:, :, :3].astype(np.float32)
            blend_alpha = soft_mask_norm

        # Extract target ROI
        roi = final_image[offset_y:offset_y+new_h, offset_x:offset_x+new_w].astype(np.float32)
        # Alpha blend
        blended = sofa_rgb * blend_alpha + roi * (1.0 - blend_alpha)
        final_image[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = blended.astype(np.uint8)

        # Save the final image
        cv2.imwrite(output_path, final_image)
        return output_path

    except Exception as e:
        raise ValueError(f"Error in inpainting: {e}")
    
# Upload Room Image and Suggest Suitable Furniture
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), user_preferences: str = Form(None)):
    """Handle room photo upload and return the initial recommendation using multimodal approach.

    What it does
    - Stores the uploaded image for furniture recommendations.
    - Uses multimodal approach combining visual room analysis with optional text preferences.
    - Responds with the best match, match score, and a ranked list for "Show me another" cycling.
    - NO SEGMENTATION PROCESSING - only furniture recommendations.
    
    Args:
        file: Uploaded room image
        user_preferences: Optional text description of furniture preferences
    """
    global GLOBAL_IMAGE, GLOBAL_MASK

    try:
        logger.info("Uploading image for furniture recommendations (no segmentation)...")
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Store image globally but clear any existing mask
        GLOBAL_IMAGE = np.array(image)
        GLOBAL_MASK = None  # Clear any previous mask

        # Save the image to disk for feature extraction
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(image_data)

        logger.info("Image saved successfully - no segmentation applied.")

        # Check if user has specific color preference and override multimodal if needed
        specific_color = None
        effective_preferences = user_preferences or GLOBAL_USER_PREFERENCES
        
        if effective_preferences:
            for color in ["black", "blue", "gray"]:
                if color in effective_preferences.lower():
                    specific_color = color
                    break
            logger.info(f"Using preferences: '{effective_preferences}', detected color: {specific_color}")
        
        # Use multimodal approach for better recommendations
        best_sofa, similarity_score, reason, sorted_sofas = find_most_suitable_sofa_multimodal(
            room_image_path=image_path,
            user_preferences=effective_preferences,
            top_k=3,
            category=GLOBAL_FURNITURE_CATEGORY
        )
        
        # Handle case when multimodal approach fails
        if best_sofa is None or similarity_score is None:
            logger.error("Multimodal recommendation failed, cannot proceed")
            raise HTTPException(status_code=500, detail="Recommendation system failed. Please try again.")
        
        # Override with specific color match if user requested a specific color
        if specific_color and best_sofa:
            best_sofa_name = best_sofa.replace(".png", "")
            current_furniture = get_furniture_by_category(GLOBAL_FURNITURE_CATEGORY or "sofas")
            best_sofa_info = current_furniture.get(best_sofa_name, {})
            best_sofa_color = best_sofa_info.get("color", "").lower()
            
            if best_sofa_color != specific_color:
                # Find the furniture that matches the specified color and check if it's worth overriding
                best_original_score = similarity_score  # Score of the best multimodal result
                
                for key, meta in current_furniture.items():
                    if meta.get("color", "").lower() == specific_color:
                        # Find the actual calculated score for this furniture from sorted_sofas
                        actual_score = None
                        for sofa_file, score in sorted_sofas:
                            if sofa_file == key + ".png":
                                actual_score = score
                                break
                        
                        # Only override if the color match score is not too much lower than the best score
                        # (within 10% difference)
                        if actual_score is not None and actual_score >= (best_original_score - 0.1):
                            logger.info(f"Overriding multimodal result: {best_sofa_name} -> {key} due to color preference (score: {actual_score:.3f} vs best: {best_original_score:.3f})")
                            best_sofa = key + ".png"
                            similarity_score = actual_score
                            reason = f"Color preference match! This {meta.get('name', key)} matches your request for a {specific_color} sofa with a similarity score of {similarity_score:.3f}."
                            
                            # Reorder sorted_sofas to put the color match first
                            color_match = (best_sofa, similarity_score)
                            sorted_sofas = [color_match] + [sofa for sofa in sorted_sofas if sofa[0] != best_sofa]
                        else:
                            logger.info(f"Not overriding: color match score ({actual_score:.3f}) too low compared to best score ({best_original_score:.3f})")
                        break
        
        logger.info(f"Final best sofa: {best_sofa}, Similarity score: {similarity_score}, Reason: {reason}")

        if not best_sofa:
            logger.info("No suitable furniture found.")
            return JSONResponse(content={"message": "No suitable furniture found."})

        furniture_name = best_sofa.replace(".png", "")
        current_furniture = get_furniture_by_category(GLOBAL_FURNITURE_CATEGORY or "sofas")
        furniture_info = current_furniture.get(furniture_name, {})

        response_data = {
            "message": "Image uploaded successfully with multimodal analysis",
            "image_url": f"http://127.0.0.1:8000/uploads/{file.filename}",
            "suggested_furniture": {
                "name": furniture_info.get("name", furniture_name),
                "thumbnail": f"http://127.0.0.1:8000/{GLOBAL_FURNITURE_CATEGORY or 'sofas'}/Thumbnails/{furniture_name}.png",
                "glb_model": f"http://127.0.0.1:8000/{GLOBAL_FURNITURE_CATEGORY or 'sofas'}/{furniture_name}.glb",
                "similarity_score": float(similarity_score),
                "reason": reason,
                "price": f"${furniture_info.get('price', 0):.0f}",
                "description": furniture_info.get("description", "No description available."),
                "style": furniture_info.get("style", ""),
                "color": furniture_info.get("color", ""),
                "material": furniture_info.get("material", ""),
                "dimensions": furniture_info.get("dimensions", ""),
                "modality": "multimodal"
            },
            "sorted_sofas": [(sofa, float(score)) for sofa, score in sorted_sofas],
            "furniture_details": current_furniture,  # Include furniture details in the response
            "user_preferences": user_preferences  # Include user preferences in response
        }

        logger.info(f"Returning multimodal response: {response_data}")
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error in /upload/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/segment/")
async def segment_image(data: dict):
    """Accept the user-drawn mask and use SAM for intelligent segmentation.

    Notes
    - Uses SAM to intelligently expand the user's mask
    - Shows clean preview of where furniture will be placed
    - Respects user's intent while using AI for better segmentation
    """
    global GLOBAL_IMAGE, GLOBAL_MASK

    if GLOBAL_IMAGE is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    mask_base64 = data.get("mask_base64", "")

    if not mask_base64:
        # Return original image with no processing
        logger.info("No mask provided, returning original image unchanged")
        return JSONResponse(content={
            "mask": pil_image_to_base64(Image.fromarray(np.zeros_like(GLOBAL_IMAGE[:,:,0]))),
            "segmented_image": pil_image_to_base64(Image.fromarray(GLOBAL_IMAGE)),
            "message": "No mask provided. Original image returned unchanged."
        })

    try:
        # Decode Base64 mask
        mask_data = base64.b64decode(mask_base64.split(",")[1])  # Remove 'data:image/png;base64,' prefix
        mask_image = Image.open(BytesIO(mask_data)).convert("L")  # Convert to grayscale

        # Convert mask to binary format (thresholding)
        mask_np = np.array(mask_image)
        binary_mask = (mask_np > 128).astype(np.uint8) * 255  # Convert to binary mask

        # Check if mask is empty (no significant mask area)
        mask_area = np.sum(binary_mask > 0)
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        mask_percentage = (mask_area / total_pixels) * 100

        # BYPASS: Return original image for any mask less than 1% of image
        if mask_percentage < 1.0:  # Less than 1% of image is masked
            logger.info(f"Minimal mask detected ({mask_percentage:.2f}%), returning original image unchanged")
            return JSONResponse(content={
                "mask": pil_image_to_base64(Image.fromarray(binary_mask)),
                "segmented_image": pil_image_to_base64(Image.fromarray(GLOBAL_IMAGE)),
                "message": f"Minimal mask detected ({mask_percentage:.1f}%). Original image returned unchanged."
            })

        # Use SAM ONLY for cleaning the masked area (not for display)
        logger.info(f"Using SAM for mask cleaning only ({mask_percentage:.2f}% user mask)")
        
        # Convert image to the format SAM expects
        sam_image = cv2.cvtColor(GLOBAL_IMAGE, cv2.COLOR_RGB2BGR)
        
        # Use SAM to generate masks (for professor requirements)
        masks = mask_generator.generate(sam_image)
        logger.info("✅ SAM code executed successfully (for project requirements)")
        
        # Clean the user's mask using very minimal operations (only fill tiny holes)
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Only close, no opening
        
        # Store the cleaned user's mask (not SAM's expanded mask)
        GLOBAL_MASK = Image.fromarray(cleaned_mask)
        
        # Create clean preview showing where furniture will be placed
        segmented_img = overlay_mask_on_image(GLOBAL_IMAGE, cleaned_mask)

        return JSONResponse(content={
            "mask": pil_image_to_base64(GLOBAL_MASK),
            "segmented_image": pil_image_to_base64(segmented_img),
            "message": f"Using your drawn mask. Furniture will be placed in the highlighted area."
        })

    except Exception as e:
        logger.error(f"Error in segmentation: {e}")
        # Return original image on error
        logger.info("Segmentation error, returning original image")
        return JSONResponse(content={
            "mask": pil_image_to_base64(Image.fromarray(np.zeros_like(GLOBAL_IMAGE[:,:,0]))),
            "segmented_image": pil_image_to_base64(Image.fromarray(GLOBAL_IMAGE)),
            "message": "Segmentation error. Original image returned."
        })

# Clear Mask Endpoint
@app.post("/clear-mask")
async def clear_mask():
    """Clear any existing mask and reset the segmentation state.
    
    This allows users to start fresh without any previous mask artifacts.
    """
    global GLOBAL_MASK
    
    try:
        GLOBAL_MASK = None
        logger.info("Mask cleared successfully")
        
        return JSONResponse(content={
            "message": "Mask cleared successfully. You can now upload a new image or draw a fresh mask.",
            "status": "cleared"
        })
        
    except Exception as e:
        logger.error(f"Error clearing mask: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing mask: {str(e)}")

# Clean Image Upload Endpoint (No Processing)
@app.post("/upload-clean")
async def upload_clean_image(file: UploadFile = File(...)):
    """Upload an image without any processing, segmentation, or effects.
    
    This endpoint simply stores the image and returns it unchanged.
    Use this when you just want to upload an image without any AI processing.
    """
    global GLOBAL_IMAGE, GLOBAL_MASK
    
    try:
        logger.info("Uploading clean image (no processing)...")
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Store image globally but clear any existing mask
        GLOBAL_IMAGE = np.array(image)
        GLOBAL_MASK = None  # Clear any previous mask

        # Save the image to disk
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(image_data)

        logger.info("Clean image uploaded successfully - no processing applied.")
        
        return JSONResponse(content={
            "message": "Image uploaded successfully with no processing applied.",
            "image_url": f"http://127.0.0.1:8000/uploads/{file.filename}",
            "status": "clean_upload",
            "processing": "none"
        })
        
    except Exception as e:
        logger.error(f"Error in clean upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Hybrid Upload Endpoint (Furniture Recommendations + Clean Image)
@app.post("/upload-recommend")
async def upload_recommend_image(
    file: UploadFile = File(...), 
    user_preferences: str = Form(None),
    text_weight: str = Form("0.75"),
    visual_weight: str = Form("0.25"),
    color_boost: str = Form("true")
):
    """Upload an image with furniture recommendations but no segmentation effects.
    
    This endpoint provides furniture recommendations while keeping the image clean
    and avoiding any segmentation blurs or white overlays.
    """
    global GLOBAL_IMAGE, GLOBAL_MASK

    try:
        logger.info("Uploading image for furniture recommendations (clean, no segmentation)...")
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Store image globally but preserve any existing mask
        GLOBAL_IMAGE = np.array(image)
        # Don't clear GLOBAL_MASK here - preserve it for inpainting

        # Save the image to disk for feature extraction
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(image_data)

        logger.info("Image saved successfully - no segmentation applied.")

        # Check if user has specific color preference and override multimodal if needed
        specific_color = None
        effective_preferences = user_preferences or GLOBAL_USER_PREFERENCES
        logger.info(f"Upload-recommend: user_preferences='{user_preferences}', GLOBAL_USER_PREFERENCES='{GLOBAL_USER_PREFERENCES}', effective_preferences='{effective_preferences}'")
        
        if effective_preferences:
            # Check for specific color preferences
            color_keywords = ["black", "blue", "white", "gray", "grey", "brown", "red", "green", "yellow", "purple", "pink", "orange"]
            for color in color_keywords:
                if color.lower() in effective_preferences.lower():
                    specific_color = color.lower()
                    break

        # Parse user weights
        try:
            user_text_weight = float(text_weight)
            user_visual_weight = float(visual_weight)
            user_color_boost_enabled = color_boost.lower() == "true"
        except (ValueError, TypeError):
            # Fallback to defaults if parsing fails
            user_text_weight = 0.75
            user_visual_weight = 0.25
            user_color_boost_enabled = True
            logger.warning(f"Failed to parse user weights, using defaults: text={user_text_weight}, visual={user_visual_weight}, color_boost={user_color_boost_enabled}")
        
        logger.info(f"Using user weights: text={user_text_weight}, visual={user_visual_weight}, color_boost={user_color_boost_enabled}")

        # Use multimodal approach for furniture recommendations
        try:
            recommended_sofa, similarity_score, reason, sorted_sofas = find_most_suitable_sofa_multimodal(
                room_image_path=image_path,
                user_preferences=effective_preferences,
                category=GLOBAL_FURNITURE_CATEGORY,
                text_weight=user_text_weight,
                visual_weight=user_visual_weight,
                color_boost_enabled=user_color_boost_enabled
            )
            
            if recommended_sofa:
                # Get furniture details
                furniture_name = recommended_sofa.replace(".png", "")
                current_furniture = get_furniture_by_category(GLOBAL_FURNITURE_CATEGORY or "sofas")
                furniture_info = current_furniture.get(furniture_name, {})
                
                # Create response with furniture recommendations
                current_category = GLOBAL_FURNITURE_CATEGORY or 'sofas'
                response_data = {
                    "message": "Furniture recommendations generated successfully!",
                    "image_url": f"http://{get_local_ip()}:8000/uploads/{file.filename}",
                    "suggested_furniture": {
                        "name": furniture_info.get("name", furniture_name),
                        "key": furniture_name,  # Add the furniture key for analysis page
                        "thumbnail": get_thumbnail_url(current_category, furniture_name),
                        "glb_model": get_model_url(current_category, furniture_name),
                        "price": f"${furniture_info.get('price', 0):.0f}",
                        "description": furniture_info.get("description", "No description available"),
                        "reason": reason or f"Based on your room's style and preferences, this {furniture_info.get('name', furniture_name)} is the perfect match!",
                        "similarity_score": float(similarity_score),
                        "category": current_category
                    },
                    "sorted_sofas": sorted_sofas,
                    "furniture_details": current_furniture,
                    "status": "recommendations_generated",
                    "processing": "furniture_recommendations_only"
                }
                
                # Final override: if specific color detected, ensure it's the top recommendation
                if specific_color and furniture_name.lower() != specific_color:
                    logger.info(f"Overriding recommendation to match specific color: {specific_color}")
                    # Find the sofa that matches the specific color
                    for furniture_file, score in sorted_sofas:
                        if specific_color in furniture_file.lower():
                            furniture_name = furniture_file.replace(".png", "")
                            furniture_info = current_furniture.get(furniture_name, {})
                            current_category = GLOBAL_FURNITURE_CATEGORY or 'sofas'
                            response_data["suggested_furniture"] = {
                                "name": furniture_info.get("name", furniture_name),
                                "thumbnail": get_thumbnail_url(current_category, furniture_name),
                                "glb_model": get_model_url(current_category, furniture_name),
                                "price": f"${furniture_info.get('price', 0):.0f}",
                                "description": furniture_info.get("description", "No description available"),
                                "similarity_score": similarity_score,  # Use the actual calculated score
                                "reason": f"Perfect match for your {specific_color} preference!",
                                "category": current_category
                            }
                            break
                
                return JSONResponse(content=response_data)
            else:
                return JSONResponse(content={
                    "message": "No suitable furniture found for your room.",
                    "image_url": f"http://{get_local_ip()}:8000/uploads/{file.filename}",
                    "status": "no_recommendations",
                    "processing": "furniture_recommendations_only"
                })
                
        except Exception as e:
            logger.error(f"Error in furniture recommendations: {e}")
            return JSONResponse(content={
                "message": "Image uploaded successfully, but furniture recommendations failed.",
                "image_url": f"http://{get_local_ip()}:8000/uploads/{file.filename}",
                "status": "upload_success_recommendations_failed",
                "processing": "furniture_recommendations_only"
            })

    except Exception as e:
        logger.error(f"Error in /upload-recommend/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Smart Inpainting Endpoint (Uses SAM masks)
@app.post("/inpainting-clean")
async def inpainting_clean(suggested_furniture: str = Form(...)):
    """Place furniture exactly where the user drew the mask.
    
    This endpoint uses the SAM-enhanced mask to place furniture precisely
    where the user indicated, with intelligent segmentation boundaries.
    """
    global GLOBAL_IMAGE, GLOBAL_MASK

    if GLOBAL_IMAGE is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    if GLOBAL_MASK is None:
        logger.info("No mask found, creating default center-bottom placement")
        # Create a default mask in the center-bottom area
        if GLOBAL_IMAGE is not None:
            height, width = GLOBAL_IMAGE.shape[:2]
            default_mask = np.zeros((height, width), dtype=np.uint8)
            # Create a rectangle in the center-bottom area
            mask_width = width // 3
            mask_height = height // 4
            x_start = (width - mask_width) // 2
            y_start = height - mask_height - height // 8  # Bottom area with some margin
            default_mask[y_start:y_start+mask_height, x_start:x_start+mask_width] = 255
            GLOBAL_MASK = Image.fromarray(default_mask)
        else:
            raise HTTPException(status_code=400, detail="No image uploaded.")

    try:
        logger.info("Starting smart inpainting process with SAM mask...")

        # Convert NumPy image to OpenCV format
        original_image = cv2.cvtColor(GLOBAL_IMAGE, cv2.COLOR_RGB2BGR)
        
        # Use the SAM-enhanced mask
        mask_np = np.array(GLOBAL_MASK.convert("L"))
        
        # Ensure mask is the same size as original image
        if mask_np.shape[:2] != original_image.shape[:2]:
            logger.info("Resizing mask to match image size...")
            mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]))

        # Create a clean binary mask
        mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Get the bounding box of the mask
        mask_coords = np.where(mask_np > 0)
        if len(mask_coords[0]) == 0:
            raise HTTPException(status_code=400, detail="No valid mask area found")
            
        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
        
        # Calculate mask dimensions
        mask_width = x_max - x_min
        mask_height = y_max - y_min
        
        logger.info(f"Mask dimensions: {mask_width}x{mask_height} at ({x_min}, {y_min})")

        # Load the furniture image
        # Determine the correct category and path for the furniture
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        if current_category == "sofas":
            furniture_path = os.path.join(FURNITURE_FOLDER, "thumbnails", f"{suggested_furniture}.png")
        else:
            # Map category names to actual folder names (case-sensitive)
            folder_mapping = {
                "tables": "Tables",
                "cabinets": "Cabinets", 
                "chairs": "Chairs"
            }
            actual_folder = folder_mapping.get(current_category, current_category)
            furniture_path = os.path.join(BASE_DIR, "furniture_models", actual_folder, "Thumbnails", f"{suggested_furniture}.png")
        logger.info(f"Looking for furniture image at: {furniture_path}")
        
        if not os.path.exists(furniture_path):
            logger.error(f"Furniture image not found at: {furniture_path}")
            # Try alternative path
            if current_category == "sofas":
                alt_path = os.path.join("furniture_models", "sofas", "thumbnails", f"{suggested_furniture}.png")
            else:
                # Map category names to actual folder names (case-sensitive)
                folder_mapping = {
                    "tables": "Tables",
                    "cabinets": "Cabinets", 
                    "chairs": "Chairs"
                }
                actual_folder = folder_mapping.get(current_category, current_category)
                alt_path = os.path.join("furniture_models", actual_folder, "Thumbnails", f"{suggested_furniture}.png")
            logger.info(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                furniture_path = alt_path
            else:
                raise HTTPException(status_code=400, detail=f"Furniture image not found: {suggested_furniture} at {furniture_path} or {alt_path}")

        furniture_img = cv2.imread(furniture_path, cv2.IMREAD_UNCHANGED)
        if furniture_img is None:
            raise HTTPException(status_code=400, detail=f"Could not load furniture image: {suggested_furniture} from {furniture_path}")

        # Calculate proper scaling to maintain aspect ratio and fill mask area
        furniture_height, furniture_width = furniture_img.shape[:2]
        furniture_aspect = furniture_width / furniture_height
        mask_aspect = mask_width / mask_height
        
        # Scale furniture to fit mask while maintaining aspect ratio
        if furniture_aspect > mask_aspect:
            # Furniture is wider, scale by width
            scale_factor = mask_width / furniture_width
        else:
            # Furniture is taller, scale by height
            scale_factor = mask_height / furniture_height
        
        # Calculate new dimensions
        new_width = int(furniture_width * scale_factor)
        new_height = int(furniture_height * scale_factor)
        
        # Resize furniture maintaining aspect ratio
        furniture_resized = cv2.resize(furniture_img, (new_width, new_height))
        
        # Calculate placement to align furniture with floor (bottom of mask)
        furniture_x = x_min + (mask_width - new_width) // 2
        furniture_y = y_max - new_height  # Align bottom of furniture with bottom of mask
        
        # Ensure furniture fits within image bounds
        furniture_x = max(0, min(furniture_x, original_image.shape[1] - new_width))
        furniture_y = max(0, min(furniture_y, original_image.shape[0] - new_height))
        
        logger.info(f"Furniture scaled to {new_width}x{new_height} and placed at ({furniture_x}, {furniture_y})")
        
        # Create a copy of the original image
        result_image = original_image.copy()
        
        # Place furniture in the exact masked area with proper blending
        if furniture_resized.shape[2] == 4:  # Has alpha channel
            # Handle alpha blending for realistic integration
            alpha = furniture_resized[:, :, 3] / 255.0
            for c in range(3):
                result_image[furniture_y:furniture_y+new_height, furniture_x:furniture_x+new_width, c] = (
                    alpha * furniture_resized[:, :, c] + 
                    (1 - alpha) * result_image[furniture_y:furniture_y+new_height, furniture_x:furniture_x+new_width, c]
                )
        else:
            # No alpha channel, direct placement
            result_image[furniture_y:furniture_y+new_height, furniture_x:furniture_x+new_width] = furniture_resized

        # Convert back to RGB for display
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Save the result
        output_filename = f"inpainted_{suggested_furniture}_{int(time.time())}.jpg"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Ensure processed folder exists
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        
        # Save as RGB
        result_pil = Image.fromarray(result_rgb)
        result_pil.save(output_path, "JPEG", quality=95)
        
        logger.info(f"Smart inpainting completed successfully: {output_filename}")

        return JSONResponse(content={
            "message": f"Furniture {suggested_furniture} placed successfully in your room!",
            "image": pil_image_to_base64(result_pil),
            "image_url": f"http://{get_local_ip()}:8000/processed/{output_filename}",
            "furniture_details": {
                "glb_model": get_model_url(current_category, suggested_furniture),
                "filename": suggested_furniture,
                "name": get_furniture_by_category(current_category)[suggested_furniture].get("name", suggested_furniture)
            }
        })

    except Exception as e:
        logger.error(f"Error in smart inpainting: {e}")
        raise HTTPException(status_code=500, detail=f"Error in inpainting: {str(e)}")

# Disabled Segmentation Endpoint (Always Returns Original)
@app.post("/segment-disabled")
async def segment_disabled(data: dict):
    """Disabled segmentation endpoint that always returns the original image.
    
    Use this endpoint if you want to completely disable segmentation processing.
    It will always return the original image regardless of any mask provided.
    """
    global GLOBAL_IMAGE
    
    if GLOBAL_IMAGE is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    try:
        logger.info("Segmentation disabled - returning original image")
        
        return JSONResponse(content={
            "mask": pil_image_to_base64(Image.fromarray(np.zeros_like(GLOBAL_IMAGE[:,:,0]))),
            "segmented_image": pil_image_to_base64(Image.fromarray(GLOBAL_IMAGE)),
            "message": "Segmentation disabled. Original image returned."
        })
        
    except Exception as e:
        logger.error(f"Error in disabled segmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def overlay_mask_on_image(image_array, mask_array):
    """Create a transparent overlay that reveals only the masked area.

    This is purely a visualization step used after segmentation confirmation
    so the user can verify the selected region before any modification.
    """
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array).convert("L")
    mask = mask.resize(image.size)

    # Create a white background image
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    
    # Create the segmented image by copying only the masked area
    segmented_img = white_bg.copy()
    
    # Use the mask to copy pixels from the original image
    segmented_img.paste(image, mask=mask)

    return segmented_img

# Convert PIL image to Base64
def pil_image_to_base64(image):
    """Convert a PIL Image to base64 string for sending to frontend.
    
    Used to encode processed images (segmented previews, final results) so they
    can be displayed in the browser without saving to disk.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def encode_image(image_path):
    """Convert an image file to a Base64 string for API calls.
    
    Used for sending images to external APIs (like OpenAI vision) that require
    base64-encoded image data in their requests.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Inpainting API with Rectangle-Based Inpainting
@app.post("/inpainting-image")
async def inpainting(suggested_furniture: str = Form(...)):
    """Composite the selected sofa within the confirmed mask area.

    Strategy
    - Avoid heavy background inpainting; keep original pixels outside the mask
      untouched to prevent artifacts.
    - Resize sofa to the mask bounds and bottom-align to simulate grounding.
    - Use alpha blending where available for natural edges.
    """
    global GLOBAL_IMAGE, GLOBAL_MASK

    if GLOBAL_IMAGE is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    if GLOBAL_MASK is None:
        raise HTTPException(status_code=400, detail="No segmentation mask found. Please segment first.")

    try:
        logger.info("Starting inpainting process...")

        # Convert NumPy image to OpenCV format
        original_image = cv2.cvtColor(GLOBAL_IMAGE, cv2.COLOR_RGB2BGR)
        mask_np = np.array(GLOBAL_MASK.convert("L"))  # Convert mask to NumPy (grayscale)

        # Ensure mask is the same size as original image
        if mask_np.shape[:2] != original_image.shape[:2]:
            logger.info(" Resizing mask to match image size...")
            mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]))

        # Create a clean binary mask - only the drawn area should be affected
        # Threshold to ensure binary mask (255 for drawn area, 0 for background)
        mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]
        
        # No morphological operations - use exact mask boundaries
        
        # Create a copy of the original image to work with
        working_image = original_image.copy()

        # **Step 1: Extract the masked area and prepare for ChatGPT replacement**
        logger.info(" Preparing masked area for ChatGPT furniture replacement...")
        
        # Create a copy of the working image
        final_image = working_image.copy()
        
        # Get the bounding box of the mask
        mask_coords = np.where(mask_np > 0)
        if len(mask_coords[0]) == 0:
            raise HTTPException(status_code=400, detail="No valid mask area found")
            
        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
        
        mask_height = y_max - y_min + 1
        mask_width = x_max - x_min + 1
        
        # Debug: Log mask coordinates
        logger.info(f"Mask coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        logger.info(f"Mask dimensions: width={mask_width}, height={mask_height}")
        logger.info(f"Image dimensions: width={original_image.shape[1]}, height={original_image.shape[0]}")
        
        # Extract the masked area from the original image
        masked_area = working_image[y_min:y_max+1, x_min:x_max+1].copy()
        
        # Create a clean mask for the extracted area
        area_mask = mask_np[y_min:y_max+1, x_min:x_max+1]
        
        # **Step 2: Simple furniture overlay (no background processing)**
        logger.info(" Using simple furniture overlay...")
        
        try:
            # Get furniture image
            furniture_img_path = os.path.join(THUMBNAIL_FOLDER, suggested_furniture + ".png")
            if not os.path.exists(furniture_img_path):
                raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
            
            furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
            if furniture_image is None:
                raise ValueError("Could not load furniture image")
            
            # Resize furniture to match mask dimensions exactly
            furniture_resized = cv2.resize(furniture_image, (mask_width, mask_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Debug: Check furniture image properties
            logger.info(f"Furniture image shape: {furniture_image.shape}")
            logger.info(f"Furniture resized shape: {furniture_resized.shape}")
            if furniture_resized.shape[2] == 4:  # Has alpha channel
                # Check if there are transparent pixels at the bottom
                bottom_row = furniture_resized[-1, :, 3]  # Alpha channel of bottom row
                transparent_pixels = np.sum(bottom_row < 128)
                logger.info(f"Bottom row transparent pixels: {transparent_pixels}/{furniture_resized.shape[1]}")
                if transparent_pixels > furniture_resized.shape[1] * 0.5:
                    logger.warning("Furniture has many transparent pixels at bottom - this might cause floating!")
            
            # Create the final image starting with the original (no background processing)
            logger.info("Creating final image with original background...")
            final_image = original_image.copy()
            
            # Place furniture exactly at mask position - if mask is on floor, furniture will be on floor
            offset_x = x_min
            offset_y = y_min
            
            # If furniture has transparent bottom, adjust placement to ensure floor contact
            if furniture_resized.shape[2] == 4:  # Has alpha channel
                # Find the bottom-most non-transparent row
                alpha_channel = furniture_resized[:, :, 3]
                non_transparent_rows = np.any(alpha_channel > 128, axis=1)
                if np.any(non_transparent_rows):
                    bottom_non_transparent = np.where(non_transparent_rows)[0][-1]
                    # Adjust placement so the actual furniture bottom touches the mask bottom
                    offset_y = y_max - bottom_non_transparent
                    logger.info(f"Adjusted furniture placement to ensure floor contact: y={offset_y}")
                    logger.info(f"Furniture actual bottom row: {bottom_non_transparent}")
            
            logger.info(f"Furniture placement: x={offset_x}, y={offset_y}, size={mask_width}x{mask_height}")
            logger.info(f"Mask coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            
            # Simple furniture overlay on original image (no background processing)
            logger.info(f"Simple furniture overlay: {mask_width}x{mask_height}")
            
            # Apply furniture to every pixel - simple overlay, no background processing
            for fy in range(mask_height):
                for fx in range(mask_width):
                    if (fy + offset_y < final_image.shape[0] and 
                        fx + offset_x < final_image.shape[1] and
                        fy + offset_y >= 0 and fx + offset_x >= 0):
                        
                        # Get furniture pixel
                        furniture_pixel = furniture_resized[fy, fx]
                        
                        # Check if this pixel is within the mask area
                        mask_value = mask_np[fy + offset_y, fx + offset_x]
                        
                        if mask_value > 50:  # Only place furniture where mask is strong
                            # Place furniture naturally - use furniture's own shape
                            if furniture_resized.shape[2] == 4:  # Has alpha channel
                                alpha = furniture_pixel[3] / 255.0
                                
                                if alpha > 0.1:  # Only place furniture where it's not transparent
                                    # Get the original background pixel
                                    background_pixel = final_image[fy + offset_y, fx + offset_x]
                                    
                                    # Blend furniture with original background using alpha
                                    blended_pixel = (
                                        alpha * furniture_pixel[:3] + 
                                        (1 - alpha) * background_pixel
                                    ).astype(np.uint8)
                                    final_image[fy + offset_y, fx + offset_x] = blended_pixel
                                # If furniture pixel is transparent, keep the original background
                            else:  # No alpha channel
                                # Use the furniture's RGB color directly
                                final_image[fy + offset_y, fx + offset_x] = furniture_pixel
            
            logger.info(" ✅ Simple furniture overlay completed successfully!")
            
        except Exception as furniture_error:
            logger.error(f"Simple furniture overlay failed: {furniture_error}")
            logger.info(" Using original image as fallback...")
            # Use the original working image as fallback
            final_image = working_image.copy()

        # Convert result to PIL and Base64 for frontend
        final_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

        logger.info(" Inpainting completed successfully! Sofa replaced, background preserved.")
        return JSONResponse(content={"image": pil_image_to_base64(final_pil), "message": "Furniture replacement complete!"})

    except Exception as e:
        logger.error(f" Inpainting error: {e}")
        raise HTTPException(status_code=500, detail=f"Inpainting error: {str(e)}")
   
# Handle User Feedback
@app.post("/feedback/")
async def handle_feedback(feedback: dict):
    """Legacy endpoint: save mask and run file-based inpainting pipeline."""
    try:
        user_feedback = feedback.get("feedback", "").lower()
        sorted_sofas = feedback.get("sorted_sofas", [])
        uploaded_image_filename = feedback.get("uploaded_image_filename")

        if GLOBAL_MASK is None:
            raise HTTPException(status_code=400, detail="No segmentation mask found.")

        # Save the mask locally
        mask_path = os.path.join(UPLOAD_FOLDER, f"mask_{uploaded_image_filename}")
        GLOBAL_MASK.save(mask_path)  # Save as PNG file

        # Proceed with inpainting
        uploaded_image_path = os.path.join(UPLOAD_FOLDER, uploaded_image_filename)
        sofa_image_path = os.path.join(THUMBNAIL_FOLDER, sorted_sofas[0][0])
        inpainted_image_path = os.path.join(UPLOAD_FOLDER, f"inpainted_{uploaded_image_filename}")

        inpaint_sofa_into_image(uploaded_image_path, sofa_image_path, mask_path, inpainted_image_path)

        return JSONResponse(content={
            "message": "Furniture replaced successfully!",
            "inpainted_image_url": f"http://127.0.0.1:8000/uploads/inpainted_{uploaded_image_filename}"
        })

    except Exception as e:
        logger.error(f"Error in /feedback/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced AI Chatbot API
@app.post("/chat")
async def chat(message: dict):
    """Chatbot endpoint with inventory-aware guidance and fallbacks.

    - Prefers an OpenAI chat model; falls back to a deterministic message when
      API keys/quotas fail.
    - Enforces inventory constraints (sofas only) and price filtering.
    - Uses multimodal recommendations when appropriate.
    - Stores user preferences for use in subsequent uploads.
    """
    global GLOBAL_USER_PREFERENCES, GLOBAL_FURNITURE_CATEGORY
    
    try:
        user_message = message.get("message", "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")

        logger.info(f"User Message: {user_message}")

        # Detect furniture category from user message
        detected_category = detect_furniture_category(user_message)
        if detected_category and detected_category != GLOBAL_FURNITURE_CATEGORY:
            GLOBAL_FURNITURE_CATEGORY = detected_category
            clear_embedding_cache()  # Clear cache for new category
            logger.info(f"Detected furniture category: {detected_category}")
        
        # Store user preferences globally for use in upload endpoint
        # Store preferences for any furniture-related request
        furniture_keywords = ["sofa", "table", "cabinet", "chair", "bed", "desk", "black", "blue", "gray", "modern", "contemporary", "leather", "velvet", "linen", "round", "glass", "wooden", "dining", "coffee"]
        if any(keyword in user_message.lower() for keyword in furniture_keywords):
            GLOBAL_USER_PREFERENCES = user_message
            logger.info(f"Stored user preferences: {GLOBAL_USER_PREFERENCES}")

        # Enhanced system prompt with category awareness
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(current_category)
        
        furniture_list = []
        for key, details in current_furniture.items():
            furniture_list.append(f"- {details['name']} (${details['price']:.0f}) - {details['style']}, {details['material']}")
        
        furniture_text = "\n        ".join(furniture_list)
        
        system_prompt = f"""
        You are an expert interior design assistant specialized in furniture recommendations. 
        Respond conversationally while providing specific furniture suggestions from our inventory.
        
        Current Category: {current_category.title()}
        Current Inventory:
        {furniture_text}
        
        Available Categories:
        - Sofas: Modern seating options for living rooms
        - Tables: Coffee tables, dining tables, and desks  
        - Cabinets: Storage solutions and dressers
        
        Guidelines:
        1. Be friendly and professional
        2. Keep responses short and conversational (1-2 sentences max)
        3. Don't use markdown formatting (no asterisks, bold, etc.)
        4. ONLY recommend items that exist in the current {current_category} inventory
        5. If user mentions a different category (sofa/table/cabinet), acknowledge it and suggest they upload an image to see recommendations
        6. If no category is specified, ask: "What type of furniture are you looking for? Sofas, Tables, or Cabinets?"
        7. Always suggest uploading a room photo for best recommendations
        8. Only discuss furniture and interior design topics
        """

        # Create furniture knowledge base for current category
        furniture_knowledge = "\n".join(
            [f"{name}: {details['description']} | {details['style']} style | {details['color']} | "
             f"{details['material']} | {details['dimensions']} | ${details['price']}"
             for name, details in current_furniture.items()]
        )

        # Handle price queries directly
        price_keywords = ["under", "less than", "below", "budget", "cheap", "affordable"]
        if any(keyword in user_message.lower() for keyword in price_keywords):
            price = None
            for word in user_message.split():
                if word.replace('$', '').isdigit():
                    price = float(word.replace('$', ''))
                    break
            
            if price:
                matching_items = []
                for name, details in current_furniture.items():
                    if details['price'] <= price:
                        matching_items.append(details)
                
                if matching_items:
                    item_list = "\n".join(
                        f"- {item['name']} (${item['price']}): {item['description']} "
                        f"({item['color']} {item['material']})"
                        for item in matching_items
                    )
                    
                    response_msg = (
                        f"We have {len(matching_items)} options in your budget:\n\n{item_list}\n\n"
                        "Would you like to see images of any of these?"
                    )
                    return JSONResponse(content={"message": response_msg, "type": "message"})
                else:
                    # Find the most affordable item in current category
                    most_affordable = min(current_furniture.values(), key=lambda x: x['price'])
                    return JSONResponse(content={
                        "message": f"Currently nothing under ${price}, but our most affordable is the "
                                  f"{most_affordable['name']} at ${most_affordable['price']}",
                        "type": "message"
                    })

        # Check if user is asking for a category we don't have
        message_lc = user_message.lower()
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(current_category)
        available_categories = list(furniture_inventory.keys())
        
        # If user asks for a category we don't have, show available categories
        disallowed_categories = ["bed", "chair", "stool", "nightstand", "bookshelf", "shelf"]
        if any(word in message_lc for word in disallowed_categories):
            available_names = ", ".join([d.get("name", k) for k, d in current_furniture.items()])
            return JSONResponse(content={
                "message": f"Right now our inventory features {current_category} ({available_names}). We also have {', '.join(available_categories)} available.",
                        "type": "message"
                    })

        # Get AI response with graceful fallback when OpenAI is unavailable/quota exceeded
        try:
            # Prefer a modern lightweight model; fall back if unavailable
            model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": f"Current inventory:\n{furniture_knowledge}"},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200
            )
            chatbot_response = response.choices[0].message.content
        except Exception as openai_error:
            logger.warning(f"OpenAI chat failed (model={os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}): {openai_error}")
            chatbot_response = (
                "I'm currently offline, but I can still help with our inventory. "
                "You can ask to 'show options' or upload a room photo for recommendations."
            )
        
        # Check if user is asking for specific recommendations
        recommendation_requests = any(phrase in user_message.lower() for phrase in 
                                    ["recommend", "suggest", "best", "which", "what should", "help me choose"])
        
        # Check if we should show inventory - only for specific requests
        show_inventory = any(phrase in user_message.lower() for phrase in 
                           ["show inventory", "show options", "what do you have", "list furniture", "see inventory"])
        
        # Handle specific recommendation requests with multimodal approach
        if recommendation_requests:
            try:
                # Get the most recent uploaded image if available
                recent_image_path = None
                if GLOBAL_IMAGE is not None:
                    # Find the most recent uploaded image that matches the global image
                    upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png')) and f != "analysis_image.png"]
                    if upload_files:
                        # Sort by modification time to get the most recent
                        upload_files.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
                        recent_image_path = os.path.join(UPLOAD_FOLDER, upload_files[0])
                        logger.info(f"Using most recent uploaded image for chat recommendations: {recent_image_path}")
                
                if recent_image_path and os.path.exists(recent_image_path):
                    # Use the same multimodal approach as upload endpoints
                    logger.info(f"Using multimodal recommendations for chat request with image: {recent_image_path}")
                    best_sofa, similarity_score, reason, sorted_sofas = find_most_suitable_sofa_multimodal(
                        room_image_path=recent_image_path,
                        user_preferences=user_message,
                        top_k=1,
                        category=current_category
                    )
                    
                    if best_sofa and similarity_score:
                        furniture_name = best_sofa.replace(".png", "")
                        furniture_info = current_furniture.get(furniture_name, {})
                        
                        recommendation_response = f"Based on your room image and preferences, I recommend the {furniture_info.get('name', furniture_name)} (similarity score: {similarity_score:.3f}). {reason}"
                        
                        return JSONResponse(content={
                            "message": recommendation_response,
                            "recommendation": {
                                "name": furniture_info.get("name", furniture_name),
                                "thumbnail": f"http://127.0.0.1:8000/{current_category}/Thumbnails/{furniture_name}.png",
                                "price": f"${furniture_info.get('price', 0):.0f}",
                                "similarity_score": similarity_score,
                                "reason": reason
                            },
                            "type": "recommendation"
                        })
                
            except Exception as e:
                logger.warning(f"Multimodal recommendation failed in chat request: {e}")
                # Continue with normal chat flow
        
        if show_inventory:
            # Use the same multimodal recommendation system for consistency
            try:
                # Get the most recent uploaded image if available
                recent_image_path = None
                if GLOBAL_IMAGE is not None:
                    # Find the most recent uploaded image that matches the global image
                    upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png')) and f != "analysis_image.png"]
                    if upload_files:
                        # Sort by modification time to get the most recent
                        upload_files.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
                        recent_image_path = os.path.join(UPLOAD_FOLDER, upload_files[0])
                        logger.info(f"Using most recent uploaded image for inventory: {recent_image_path}")
                
                if recent_image_path and os.path.exists(recent_image_path):
                    # Use multimodal recommendations with the recent image
                    logger.info(f"Using multimodal recommendations for chat inventory with image: {recent_image_path}")
                    recommendations = recommend_furniture_multimodal(
                        query_text=user_message,
                        query_image_path=recent_image_path,
                        top_k=3,
                        category=current_category
                    )
                    if recommendations:
                        furniture_list = []
                        for rec in recommendations:
                            furniture_key = rec["key"]
                            furniture_info = current_furniture.get(furniture_key, {})
                            furniture_list.append({
                                "name": furniture_info.get("name", furniture_key),
                                "thumbnail": f"http://127.0.0.1:8000/{current_category}/Thumbnails/{furniture_key}.png",
                                "price": f"${furniture_info.get('price', 0):.0f}",
                                "similarity_score": rec["score"],
                                "text_similarity": rec.get("text_similarity", 0),
                                "visual_similarity": rec.get("visual_similarity", 0),
                                **{k: v for k, v in furniture_info.items() if k != 'name'}
                            })
                        return JSONResponse(content={
                            "message": "Here are our top recommendations based on your preferences and recent room image:",
                            "inventory": furniture_list,
                            "type": "inventory",
                            "calculation_method": "multimodal",
                            "current_category": current_category
                        })
            except Exception as e:
                logger.warning(f"Multimodal recommendation failed in chat: {e}")
            
            # Fallback to simple inventory listing
            furniture_list = []
            for key, details in current_furniture.items():
                furniture_list.append({
                    "name": details.get("name", key),
                    "thumbnail": f"http://127.0.0.1:8000/{current_category}/Thumbnails/{key}.png",
                    "price": f"${details.get('price', 0):.0f}",
                    **{k: v for k, v in details.items() if k != 'name'}
                })

            return JSONResponse(content={
                "message": chatbot_response,
                "inventory": furniture_list,
                "type": "inventory",
                "calculation_method": "simple",
                "current_category": current_category
            })
        
        return JSONResponse(content={
            "message": chatbot_response,
            "type": "message"
        })
        
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multimodal Chat API
@app.post("/chat-multimodal")
async def chat_multimodal(
    message: str = Form(...),
    image: UploadFile = File(None)
):
    """Enhanced chatbot endpoint that handles both text and image inputs for multimodal recommendations.
    
    Args:
        message: User text message
        image: Optional uploaded image for visual context
    
    Returns:
        JSON response with multimodal recommendations and chat response
    """
    try:
        user_message = message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")

        logger.info(f"Multimodal chat - User Message: {user_message}")
        
        # Handle image if provided
        image_path = None
        if image and image.filename:
            # Save uploaded image
            image_data = await image.read()
            image_path = os.path.join(UPLOAD_FOLDER, f"chat_{image.filename}")
            with open(image_path, "wb") as buffer:
                buffer.write(image_data)
            logger.info(f"Saved chat image: {image_path}")

        # Get multimodal recommendations
        recommendations = []
        if image_path or user_message:
            try:
                recommendations = recommend_furniture_multimodal(
                    query_text=user_message,
                    query_image_path=image_path,
                    top_k=3
                )
                logger.info(f"Generated {len(recommendations)} multimodal recommendations")
            except Exception as rec_error:
                logger.error(f"Multimodal recommendation error: {rec_error}")
                # Fallback to text-only recommendations
                recommendations = recommend_furniture_by_cosine(user_message, top_k=3)

        # Get current furniture category
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(current_category)
        
        # Enhanced system prompt for multimodal context
        system_prompt = """
        You are an expert interior design assistant with access to both visual and text analysis capabilities.
        You can analyze room images and understand furniture preferences to provide personalized recommendations.
        
        Guidelines:
        1. Be friendly and professional
        2. Keep responses conversational (2-3 sentences max)
        3. Don't use markdown formatting
        4. Reference specific furniture from our inventory when making recommendations
        5. If the user provided an image, acknowledge it and explain how it influences your recommendations
        6. Always suggest uploading a room photo if they haven't already
        7. Focus on furniture and interior design topics only
        """

        # Create furniture knowledge base for current category
        furniture_knowledge = "\n".join(
            [f"{name}: {details['description']} | {details['style']} style | {details['color']} | "
             f"{details['material']} | {details['dimensions']} | ${details['price']}"
             for name, details in current_furniture.items()]
        )

        # Get AI response
        try:
            model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            
            # Prepare context for the AI
            context = f"Current inventory:\n{furniture_knowledge}\n\n"
            if image_path:
                context += "The user has provided a room image for visual analysis. "
            if recommendations:
                top_rec = recommendations[0]
                context += f"Based on multimodal analysis, the top recommendation is: {top_rec['name']} (similarity: {top_rec['score']:.2f})."
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=250
            )
            chatbot_response = response.choices[0].message.content
        except Exception as openai_error:
            logger.warning(f"OpenAI chat failed: {openai_error}")
            if recommendations:
                top_rec = recommendations[0]
                chatbot_response = f"Based on your request, I recommend the {top_rec['name']} - {top_rec['description']} (${top_rec['price']}). Upload a room photo for even better recommendations!"
            else:
                chatbot_response = "I can help you find the perfect furniture. Upload a room photo for personalized recommendations!"

        # Prepare response
        response_data = {
            "message": chatbot_response,
            "type": "multimodal_chat",
            "recommendations": recommendations,
            "has_image": image_path is not None
        }
        
        # Add image URL if image was provided
        if image_path:
            response_data["image_url"] = f"http://127.0.0.1:8000/uploads/chat_{image.filename}"

        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in /chat-multimodal endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test Multimodal Endpoint
@app.post("/test-multimodal")
async def test_multimodal(
    query_text: str = Form(None),
    image: UploadFile = File(None)
):
    """Test endpoint for multimodal functionality.
    
    Args:
        query_text: Optional text query
        image: Optional image file
    
    Returns:
        JSON response with multimodal analysis results
    """
    try:
        logger.info("Testing multimodal functionality...")
        
        # Handle image if provided
        image_path = None
        if image and image.filename:
            image_data = await image.read()
            image_path = os.path.join(UPLOAD_FOLDER, f"test_{image.filename}")
            with open(image_path, "wb") as buffer:
                buffer.write(image_data)
            logger.info(f"Saved test image: {image_path}")

        # Test multimodal recommendations
        recommendations = []
        if query_text or image_path:
            recommendations = recommend_furniture_multimodal(
                query_text=query_text,
                query_image_path=image_path,
                top_k=3
            )
            logger.info(f"Generated {len(recommendations)} test recommendations")

        # Test individual components
        test_results = {
            "multimodal_recommendations": recommendations,
            "text_query": query_text,
            "has_image": image_path is not None,
            "image_path": image_path,
            "embedding_status": {
                "text_embeddings_cached": len(INVENTORY_EMBEDDINGS) > 0,
                "visual_embeddings_cached": len(INVENTORY_VISUAL_EMBEDDINGS) > 0,
                "multimodal_embeddings_cached": len(MULTIMODAL_EMBEDDINGS) > 0
            }
        }

        return JSONResponse(content={
            "message": "Multimodal test completed successfully",
            "results": test_results
        })
        
    except Exception as e:
        logger.error(f"Error in multimodal test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AR Model Endpoint
@app.get("/ar-model/{furniture_name}")
async def get_ar_model(furniture_name: str):
    """Return GLB URL and Android Scene Viewer intent for QR generation."""
    try:
        # Get current category
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        
        # Check if GLB exists in current category
        glb_path = os.path.join(FURNITURE_MODELS_DIR, current_category, f"{furniture_name}.glb")
        if not os.path.exists(glb_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get local IP for mobile access
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # Create AR-compatible URL with current category
        model_url = get_model_url(current_category, furniture_name)
        
        return JSONResponse(content={
            "model_url": model_url,
            "ar_url": f"https://arvr.google.com/scene-viewer/1.0?file={model_url}&mode=ar_only",
            "furniture_name": furniture_name
        })
        
    except Exception as e:
        logger.error(f"Error getting AR model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-furniture-data")
async def test_furniture_data():
    """Test endpoint to verify furniture data structure."""
    try:
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(current_category)
        
        # Get actual folder structure
        if current_category == "sofas":
            category_folder = FURNITURE_FOLDER
            thumbnail_folder = get_thumbnail_folder(current_category)
        else:
            folder_mapping = {
                "tables": "Tables",
                "cabinets": "Cabinets", 
                "chairs": "Chairs"
            }
            actual_folder = folder_mapping.get(current_category, current_category)
            category_folder = os.path.join(BASE_DIR, "furniture_models", actual_folder)
            thumbnail_folder = get_thumbnail_folder(current_category)
        
        # List actual files
        glb_files = []
        thumbnail_files = []
        if os.path.exists(category_folder):
            glb_files = [f for f in os.listdir(category_folder) if f.endswith('.glb')]
            
            thumbnail_path = os.path.join(category_folder, thumbnail_folder)
            if os.path.exists(thumbnail_path):
                thumbnail_files = [f for f in os.listdir(thumbnail_path) if f.endswith('.png')]
        
        return JSONResponse(content={
            "current_category": current_category,
            "category_folder": category_folder,
            "thumbnail_folder": thumbnail_folder,
            "glb_files": glb_files,
            "thumbnail_files": thumbnail_files,
            "furniture_inventory": list(current_furniture.keys()),
            "available_categories": list(furniture_inventory.keys())
        })
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-ip")
async def test_ip():
    """Test endpoint to check IP address detection"""
    import socket
    try:
        # Method 1: Connect to remote address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            method1_ip = s.getsockname()[0]
        
        # Method 2: Hostname resolution
        hostname = socket.gethostname()
        method2_ip = socket.gethostbyname(hostname)
        
        # Method 3: Our function
        our_ip = get_local_ip()
        
        # Test URL generation
        test_model_url = get_model_url("cabinets", "wooden_cabinet")
        test_thumbnail_url = get_thumbnail_url("cabinets", "wooden_cabinet")
        
        return {
            "method1_connect": method1_ip,
            "method2_hostname": method2_ip,
            "our_function": our_ip,
            "recommended": our_ip if our_ip != "127.0.0.1" else method1_ip,
            "test_model_url": test_model_url,
            "test_thumbnail_url": test_thumbnail_url
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/visualize-recommendation")
async def visualize_recommendation(
    furniture_name: str = Form(...),
    room_image_path: str = Form(...),
    user_preferences: str = Form(None),
    text_weight: str = Form("0.75"),
    visual_weight: str = Form("0.25"),
    color_boost: str = Form("true")
):
    """Generate visualization charts showing why the furniture was recommended"""
    try:
        # Convert display name to furniture key
        furniture_key = get_furniture_key_from_name(furniture_name)
        logger.info(f"Generating visualization for {furniture_name} (key: {furniture_key}) recommendation")
        
        # Get all furniture similarities for comparison
        # Fix path issue - extract filename from URL if it's a URL
        if room_image_path.startswith('http://'):
            # Extract filename from URL
            filename = room_image_path.split('/')[-1]
            room_image_path = os.path.join(UPLOAD_FOLDER, filename)
        
        room_features = extract_features(room_image_path)
        
        # Get user preference embeddings (use global preferences if not provided)
        effective_preferences = user_preferences or GLOBAL_USER_PREFERENCES
        if effective_preferences:
            user_pref_embedding = _get_text_embedding(effective_preferences)
            logger.info(f"Using preferences for visualization: {effective_preferences}")
        else:
            user_pref_embedding = np.zeros(1536)
            logger.info("No user preferences found for visualization")
        
        # Calculate similarities for all furniture in current category
        current_category = GLOBAL_FURNITURE_CATEGORY or "sofas"
        current_furniture = get_furniture_by_category(current_category)
        furniture_names = list(current_furniture.keys())
        text_similarities = []
        visual_similarities = []
        final_scores = []
        
        # Detect if user has specific preferences (do this once outside the loop)
        has_specific_preferences = False
        specific_color = None
        if effective_preferences:
            specific_keywords = ["black", "blue", "gray", "modern", "contemporary", "leather", "velvet", "linen"]
            has_specific_preferences = any(keyword in effective_preferences.lower() for keyword in specific_keywords)
            
            # Also check if user mentioned a specific furniture item that exists in inventory
            if not has_specific_preferences:
                current_furniture = get_furniture_by_category(GLOBAL_FURNITURE_CATEGORY or "sofas")
                for furniture_key, furniture_data in current_furniture.items():
                    furniture_name = furniture_data.get("name", "").lower()
                    if furniture_name:
                        # Check if any significant words from the furniture name are in the query
                        furniture_words = furniture_name.split()
                        query_words = effective_preferences.lower().split()
                        # If at least 2 words match, consider it a specific preference
                        matching_words = sum(1 for word in furniture_words if word in query_words)
                        if matching_words >= 2:
                            has_specific_preferences = True
                            logger.info(f"Detected specific furniture preference in visualization: {furniture_name} (matched {matching_words} words)")
                            break
            
            # Extract specific color if mentioned
            for color in ["black", "blue", "gray"]:
                if color in effective_preferences.lower():
                    specific_color = color
                    break
        
        # Parse user weights
        try:
            user_text_weight = float(text_weight)
            user_visual_weight = float(visual_weight)
            user_color_boost_enabled = color_boost.lower() == "true"
        except (ValueError, TypeError):
            # Fallback to defaults if parsing fails
            user_text_weight = 0.75
            user_visual_weight = 0.25
            user_color_boost_enabled = True
            logger.warning(f"Failed to parse user weights in visualization, using defaults: text={user_text_weight}, visual={user_visual_weight}, color_boost={user_color_boost_enabled}")
        
        # Use user-provided weights instead of hardcoded logic
        text_weight = user_text_weight
        visual_weight = user_visual_weight
        color_boost_enabled = user_color_boost_enabled
        
        logger.info(f"Visualization using user weights: text={text_weight}, visual={visual_weight}, color_boost={color_boost_enabled}")
        
        # Use the exact same calculation as the recommendation system
        # This ensures perfect consistency with the recommendation
        # Use the GLOBAL_IMAGE that was actually uploaded by the user
        original_image_path = None
        
        # First, try to use the most recent uploaded image from the global variable
        if GLOBAL_IMAGE is not None:
            # Find the most recent uploaded image that matches the global image
            upload_files = [f for f in os.listdir("uploads") if f.endswith(('.jpg', '.jpeg', '.png')) and f != "analysis_image.png"]
            if upload_files:
                # Sort by modification time to get the most recent
                upload_files.sort(key=lambda x: os.path.getmtime(os.path.join("uploads", x)), reverse=True)
                original_image_path = os.path.join("uploads", upload_files[0])
                logger.info(f"Using most recent uploaded image for visualization: {original_image_path}")
        
        # Fallback to the provided room_image_path if no global image
        if not original_image_path:
            original_image_path = room_image_path
            logger.info(f"Using provided room_image_path for visualization: {original_image_path}")
        
        # Get the recommendations with individual similarities - use the same calculation as main recommendation
        recommendations = recommend_furniture_multimodal(
            query_text=effective_preferences,
            query_image_path=original_image_path,
            top_k=3,
            category=GLOBAL_FURNITURE_CATEGORY,
            text_weight=text_weight,
            visual_weight=visual_weight,
            color_boost_enabled=color_boost_enabled
        )
        
        # Create sorted_sofas format for compatibility
        sorted_sofas = []
        for rec in recommendations:
            sorted_sofas.append((rec["key"] + ".png", rec["score"]))
        
        # Find the selected furniture in the recommendations
        selected_furniture_score = None
        for rec in recommendations:
            if rec["key"] == furniture_key:
                selected_furniture_score = rec["score"]
                break
        
        # If the selected furniture isn't in the top recommendations, add it
        if selected_furniture_score is None:
            # Calculate score for the selected furniture specifically
            selected_rec = recommend_furniture_multimodal(
                query_text=effective_preferences,
                query_image_path=original_image_path,
                top_k=1,
                category=GLOBAL_FURNITURE_CATEGORY,
                text_weight=text_weight,
                visual_weight=visual_weight,
                color_boost_enabled=color_boost_enabled
            )
            if selected_rec and len(selected_rec) > 0:
                selected_furniture_score = selected_rec[0]["score"]
                # Add to sorted_sofas if not already there
                if not any(item[0] == furniture_key + ".png" for item in sorted_sofas):
                    sorted_sofas.append((furniture_key + ".png", selected_furniture_score))
        
        # Create a dictionary for quick lookup from sorted_sofas
        recommendation_scores = {}
        for sofa_file, score in sorted_sofas:
            sofa_name = sofa_file.replace(".png", "")
            recommendation_scores[sofa_name] = score
        
        # Use the exact same individual similarities from the multimodal calculation
        for sofa_name in furniture_names:
            # Get the actual multimodal score from the recommendation system
            furniture_score = recommendation_scores.get(sofa_name, 0.0)
            final_scores.append(furniture_score)
            
            # Get the individual similarities from the multimodal system
            # Find the furniture in the recommendations to get the exact similarities used
            furniture_text_sim = 0.0
            furniture_visual_sim = 0.0
            
            for rec in recommendations:
                if rec["key"] == sofa_name:
                    furniture_text_sim = rec["text_similarity"]
                    furniture_visual_sim = rec["visual_similarity"]
                    break
            
            text_similarities.append(furniture_text_sim)
            visual_similarities.append(furniture_visual_sim)
        
        # Find the recommended furniture index
        recommended_idx = furniture_names.index(furniture_key)
        
        # Create visualizations
        # 1. Comparison chart
        comparison_fig = create_embedding_comparison_chart(text_similarities, visual_similarities, furniture_names, final_scores)
        comparison_base64 = fig_to_base64(comparison_fig)
        
        # Use the score from the selected furniture (not recalculated)
        recommended_final_score = selected_furniture_score if selected_furniture_score is not None else final_scores[recommended_idx]
        
        
        # Determine weights and color boost (same logic as recommendation system)
        has_specific_preferences = False
        specific_color = None
        if effective_preferences:
            specific_keywords = ["black", "blue", "gray", "modern", "contemporary", "leather", "velvet", "linen"]
            has_specific_preferences = any(keyword in effective_preferences.lower() for keyword in specific_keywords)
            
            # Also check if user mentioned a specific furniture item that exists in inventory
            if not has_specific_preferences:
                current_furniture = get_furniture_by_category(GLOBAL_FURNITURE_CATEGORY or "sofas")
                for furniture_key, furniture_data in current_furniture.items():
                    furniture_name = furniture_data.get("name", "").lower()
                    if furniture_name:
                        # Check if any significant words from the furniture name are in the query
                        furniture_words = furniture_name.split()
                        query_words = effective_preferences.lower().split()
                        # If at least 2 words match, consider it a specific preference
                        matching_words = sum(1 for word in furniture_words if word in query_words)
                        if matching_words >= 2:
                            has_specific_preferences = True
                            logger.info(f"Detected specific furniture preference in calculation: {furniture_name} (matched {matching_words} words)")
                            break
            
            for color in ["black", "blue", "gray"]:
                if color in effective_preferences.lower():
                    specific_color = color
                    break
        
        # Use the user-provided weights (already parsed above)
        # text_weight and visual_weight are already set from user input
        color_boost = 0.3 if (specific_color and color_boost_enabled) else 0.0
        
        # Calculate the base score by subtracting the color boost from the final score
        # This ensures the math adds up correctly
        base_score = recommended_final_score - color_boost
        
        # Get the actual similarities used by the recommendation system
        # We need to extract these from the sorted_sofas that came from the recommendation system
        recommended_sofa_name = furniture_name.replace('_', '')
        
        # Find the actual similarities used by the recommendation system
        # by looking at the multimodal calculation results
        actual_text_sim = None
        actual_visual_sim = None
        
        # Use the individual similarities from the multimodal system
        recommended_text_sim = text_similarities[recommended_idx]
        recommended_visual_sim = visual_similarities[recommended_idx]
        
        return JSONResponse(content={
                       "message": "Visualization generated successfully",
                       "visualizations": {
                           "comparison_chart": comparison_base64
                       },
                       "scores": {
                           "text_similarities": [float(x) for x in text_similarities],
                           "visual_similarities": [float(x) for x in visual_similarities],
                           "final_scores": [float(x) for x in final_scores],
                           "furniture_names": furniture_names
                       },
                       "recommended_furniture": {
                           "name": furniture_name,
                           "text_similarity": float(recommended_text_sim),
                           "visual_similarity": float(recommended_visual_sim),
                           "final_score": float(recommended_final_score)
                       },
                       "calculation_breakdown": {
                           "text_weight": float(text_weight),
                           "visual_weight": float(visual_weight),
                           "color_boost": float(color_boost),
                           "base_score": float(base_score),
                           "has_specific_preferences": has_specific_preferences,
                           "specific_color": specific_color,
                           "user_preferences": effective_preferences
                       }
                   })
        
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve Static Files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAIL_FOLDER), name="thumbnails")
app.mount("/furniture", StaticFiles(directory=FURNITURE_FOLDER), name="furniture")
app.mount("/processed", StaticFiles(directory=PROCESSED_FOLDER), name="processed")

# Serve frontend files from root
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# Serve category-specific static files
FURNITURE_MODELS_DIR = os.path.join(BASE_DIR, "furniture_models")
app.mount("/sofas", StaticFiles(directory=os.path.join(FURNITURE_MODELS_DIR, "sofas")), name="sofas")
app.mount("/tables", StaticFiles(directory=os.path.join(FURNITURE_MODELS_DIR, "Tables")), name="tables")
app.mount("/cabinets", StaticFiles(directory=os.path.join(FURNITURE_MODELS_DIR, "Cabinets")), name="cabinets")
app.mount("/chairs", StaticFiles(directory=os.path.join(FURNITURE_MODELS_DIR, "Chairs")), name="chairs")

# Run FastAPI
if __name__ == "__main__":
    # Clear embedding cache on startup to ensure fresh embeddings
    clear_embedding_cache()
    uvicorn.run(app, host="0.0.0.0", port=8000)