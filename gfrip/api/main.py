"""
FastAPI Application for GFRIP
Provides RESTful API endpoints for risk analysis and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
import logging
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# Import core components
from gfrip.data.ingestion import AlternativeDataIngestionPipeline
from gfrip.analytics.network_analysis import ContagionRiskAnalyzer, FinancialNetworkBuilder
from gfrip.models.risk_models import MultiModalRiskTransformer, SovereignRiskPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configurations
SECRET_KEY = "your-secret-key-here"  # In production, use environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Mock user database (in production, use a real database)
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    }
}

# Security utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict
    return None

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Pydantic models for request/response validation
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Institution(BaseModel):
    institution_id: str
    name: str
    type: str  # e.g., "bank", "insurance", "hedge_fund"
    country: str
    assets: float
    capital: float
    risk_weight: float = 1.0

class ExposureMatrix(BaseModel):
    matrix: List[List[float]]
    institution_ids: List[str]

class ShockScenario(BaseModel):
    name: str
    description: str
    node_attribute: Optional[str] = None
    edge_attribute: Optional[str] = None
    node_indices: Optional[List[int]] = None
    edge_indices: Optional[List[tuple[int, int]]] = None
    value: float

class ContagionAnalysisRequest(BaseModel):
    institutions: List[Institution]
    exposures: ExposureMatrix
    shock_scenarios: Optional[List[ShockScenario]] = []

class ContagionAnalysisResponse(BaseModel):
    baseline_risk: float
    scenario_results: Dict[str, Dict[str, Any]]
    network_metrics: Dict[str, Any]
    systemically_important_nodes: List[Dict[str, Any]]

# Initialize FastAPI app
app = FastAPI(
    title="Global Financial Risk Intelligence Platform (GFRIP) API",
    description="Advanced financial risk analysis and monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize core components
network_builder = FinancialNetworkBuilder()
contagion_analyzer = ContagionRiskAnalyzer()
data_pipeline = AlternativeDataIngestionPipeline()

# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Risk Analysis Endpoints
@app.post("/api/v1/analyze/contagion", 
          response_model=ContagionAnalysisResponse,
          summary="Analyze contagion risk in financial network")
async def analyze_contagion_risk(
    request: ContagionAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Perform contagion risk analysis on a financial network
    
    - **institutions**: List of financial institutions with their attributes
    - **exposures**: Matrix of financial exposures between institutions
    - **shock_scenarios**: Optional list of stress test scenarios to apply
    """
    try:
        # Convert institutions to DataFrame
        institutions_df = pd.DataFrame([i.dict() for i in request.institutions])
        
        # Get exposure matrix
        exposure_matrix = request.exposures.matrix
        
        # Run analysis
        results = contagion_analyzer.stress_test(
            institutions_data=institutions_df,
            exposures_matrix=exposure_matrix,
            shock_scenarios=[s.dict() for s in (request.shock_scenarios or [])]
        )
        
        # Format response
        baseline = results.pop('baseline')
        return ContagionAnalysisResponse(
            baseline_risk=baseline['systemic_risk_score'],
            scenario_results={
                name: {
                    'systemic_risk': result['systemic_risk_score'],
                    'risk_change': result.get('risk_change', 0.0),
                    'top_risky_nodes': [
                        {'node_id': node_id, 'score': float(score)}
                        for node_id, score in result.get('systemically_important_nodes', [])[:5]
                    ]
                }
                for name, result in results.items()
            },
            network_metrics={
                'num_nodes': baseline['network_metrics']['num_nodes'],
                'num_edges': baseline['network_metrics']['num_edges'],
                'density': baseline['network_metrics']['density'],
                'is_connected': baseline['network_metrics'].get('is_connected', True)
            },
            systemically_important_nodes=[
                {'node_id': node_id, 'score': float(score)}
                for node_id, score in baseline['systemically_important_nodes'][:10]
            ]
        )
        
    except Exception as e:
        logger.exception("Error in contagion risk analysis")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Example protected endpoint
@app.get("/api/v1/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
