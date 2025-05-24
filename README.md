# Global Financial Risk Intelligence Platform (GFRIP)

Advanced AI-powered risk intelligence for financial institutions, integrating alternative data, network analysis, and machine learning for comprehensive risk assessment.

## Features

- **Multi-Modal Data Integration**: Ingest and process satellite imagery, social sentiment, IoT sensors, and traditional financial data
- **Advanced Network Analysis**: Graph-based modeling of financial contagion and systemic risk
- **Machine Learning Models**: State-of-the-art deep learning for risk prediction and scenario analysis
- **Real-time Monitoring**: Continuous risk assessment with alerting capabilities
- **Regulatory Compliance**: Built-in support for stress testing and regulatory reporting

## Architecture

```
gfrip/
├── data/                    # Data ingestion and processing
│   └── ingestion.py         # Alternative data pipeline
├── models/                  # Machine learning models
│   └── risk_models.py       # Risk prediction models
├── analytics/               # Analytical modules
│   └── network_analysis.py  # Network analysis and contagion modeling
├── api/                     # FastAPI application
│   └── main.py              # REST API endpoints
├── utils/                   # Utility functions
│   └── config.py            # Configuration management
└── __init__.py              # Package initialization
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- (Optional) CUDA-enabled GPU for accelerated model training

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gfrip.git
   cd gfrip
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch with GPU support (recommended):
   ```bash
   # For CUDA 11.7
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   
   # For CPU-only
   # pip install torch torchvision torchaudio
   ```

5. Install PyTorch Geometric:
   ```bash
   # Follow instructions at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
   # Example for CUDA 11.7:
   pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
   ```

## Usage

### Running the API Server

```bash
uvicorn gfrip.api.main:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Example API Request

Analyze contagion risk:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/analyze/contagion' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "institutions": [
      {
        "institution_id": "bank1",
        "name": "Global Bank",
        "type": "bank",
        "country": "US",
        "assets": 1000000000,
        "capital": 100000000,
        "risk_weight": 1.0
      },
      {
        "institution_id": "bank2",
        "name": "International Bank",
        "type": "bank",
        "country": "UK",
        "assets": 800000000,
        "capital": 80000000,
        "risk_weight": 1.2
      }
    ],
    "exposures": {
      "matrix": [
        [0, 5000000],
        [3000000, 0]
      ],
      "institution_ids": ["bank1", "bank2"]
    },
    "shock_scenarios": [
      {
        "name": "liquidity_shock",
        "description": "10% liquidity shock to bank1",
        "node_attribute": "capital",
        "node_indices": [0],
        "value": -0.1
      }
    ]
  }'
```

## Configuration

Create a `config.yaml` file in the project root or in `gfrip/config/` to override default settings:

```yaml
environment: "production"

api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  secret_key: "your-secret-key-here"

logging:
  level: "INFO"
  file: "logs/gfrip.log"
  max_size_mb: 100
  backup_count: 5

data_sources:
  satellite:
    name: "satellite"
    type: "api"
    enabled: true
    params:
      api_key: "your-api-key"
      provider: "planet"

models:
  contagion_gnn:
    name: "contagion_gnn"
    type: "gnn"
    path: "models/contagion_gnn.pt"
    enabled: true
```

## Development

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- mypy for static type checking
- pylint for code quality

Run the following commands before committing:

```bash
black .
isort .
mypy .
pylint gfrip/
```

### Testing

Run tests with pytest:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch, PyTorch Geometric, and FastAPI
- Inspired by the latest research in financial network analysis and systemic risk modeling
