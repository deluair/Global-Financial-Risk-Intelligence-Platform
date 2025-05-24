"""
Advanced Risk Models for GFRIP
Including Financial Contagion GNN and Multi-Modal Risk Transformer
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class FinancialContagionGNN(nn.Module):
    """
    Advanced Graph Neural Network for financial contagion analysis
    Implements heterogeneous graph attention with temporal dynamics
    """
    
    def __init__(self, 
                 node_features: int = 128, 
                 edge_features: int = 64,
                 hidden_dim: int = 256, 
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 num_heads: int = 8):
        """
        Initialize the Financial Contagion GNN
        
        Args:
            node_features: Number of input node features
            edge_features: Number of input edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super(FinancialContagionGNN, self).__init__()
        
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        # Multi-layer GNN with attention mechanisms
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Contagion prediction heads
        self.contagion_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.systemic_risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_attr: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contagion risk prediction
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch vector [num_nodes]
            
        Returns:
            Dict containing node embeddings and risk predictions
        """
        # Encode nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Message passing layers
        for conv in self.convs:
            x_new = conv(x, edge_index, edge_attr)
            x = x + self.dropout(x_new)  # Residual connection
            x = self.norm(x)
        
        # Apply temporal attention if sequence data is available
        if x.dim() == 3:  # [batch_size, seq_len, hidden_dim]
            x, _ = self.temporal_attention(x, x, x)
            x = x.mean(dim=1)  # Average over sequence length
        
        # Global graph representation
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Risk predictions
        contagion_risk = self.contagion_predictor(x)
        systemic_risk = self.systemic_risk_predictor(graph_embedding)
        
        return {
            'node_embeddings': x,
            'graph_embedding': graph_embedding,
            'contagion_risk': contagion_risk,
            'systemic_risk': systemic_risk
        }


class MultiModalRiskTransformer(nn.Module):
    """
    Advanced transformer architecture for multi-modal risk synthesis
    Processes time-series, text, images, and graph data simultaneously
    """
    
    def __init__(self, 
                 d_model: int = 512, 
                 nhead: int = 8, 
                 num_layers: int = 6,
                 num_modalities: int = 4,
                 dropout: float = 0.1):
        super(MultiModalRiskTransformer, self).__init__()
        
        # Modal-specific encoders would be initialized here
        self.d_model = d_model
        self.num_modalities = num_modalities
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Main transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Risk prediction heads
        self.credit_risk_head = self._build_risk_head(d_model)
        self.market_risk_head = self._build_risk_head(d_model)
        self.operational_risk_head = self._build_risk_head(d_model)
        self.systemic_risk_head = self._build_risk_head(d_model)
        
        # Initialize weights
        self._reset_parameters()
    
    def _build_risk_head(self, input_dim: int) -> nn.Module:
        """Build a risk prediction head"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _reset_parameters(self):
        """Initialize weights with xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
               time_series_data: torch.Tensor,
               text_data: Dict[str, torch.Tensor],
               image_data: torch.Tensor,
               graph_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-modal risk prediction with attention-based fusion
        
        Args:
            time_series_data: Time-series data [batch_size, seq_len, features]
            text_data: Dictionary with text input_ids and attention_mask
            image_data: Image features [batch_size, channels, height, width]
            graph_data: Graph embeddings [batch_size, graph_embedding_dim]
            
        Returns:
            Dict containing risk predictions and attention weights
        """
        # Encode each modality (simplified - actual implementations would use specific encoders)
        # In practice, you'd have separate encoders for each modality
        batch_size = time_series_data.size(0)
        
        # Project all modalities to the same dimension
        # These are placeholders - in practice, use proper encoders
        time_series_encoded = time_series_data.mean(dim=1)  # [batch_size, d_model]
        text_encoded = text_data['pooler_output']  # Assuming pre-computed embeddings
        image_encoded = image_data.mean(dim=[1, 2])  # Global average pooling
        graph_encoded = graph_data  # Already in the right format
        
        # Stack modalities for cross-attention
        modal_stack = torch.stack([
            time_series_encoded,
            text_encoded,
            image_encoded,
            graph_encoded
        ], dim=1)  # [batch_size, num_modalities, d_model]
        
        # Cross-modal attention
        fused_representation, attention_weights = self.cross_modal_attention(
            modal_stack, modal_stack, modal_stack
        )
        
        # Flatten for transformer
        batch_size, num_modals, d_model = fused_representation.shape
        fused_flat = fused_representation.view(-1, num_modals * d_model)
        
        # Add CLS token
        cls_tokens = nn.Parameter(torch.randn(1, 1, d_model)).expand(batch_size, -1, -1)
        fused_with_cls = torch.cat([cls_tokens, fused_representation], dim=1)
        
        # Apply transformer
        risk_representation = self.fusion_transformer(fused_with_cls)
        
        # Get CLS token representation for global risk
        global_representation = risk_representation[:, 0]
        
        # Risk predictions
        return {
            'credit_risk': self.credit_risk_head(global_representation),
            'market_risk': self.market_risk_head(global_representation),
            'operational_risk': self.operational_risk_head(global_representation),
            'systemic_risk': self.systemic_risk_head(global_representation),
            'attention_weights': attention_weights
        }


class SovereignRiskPredictor(nn.Module):
    """
    Advanced sovereign debt crisis prediction using ensemble methods
    Incorporates economic indicators, alternative data, and network effects
    """
    
    def __init__(self, 
                 num_economic_features: int = 50,
                 num_network_features: int = 32,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super(SovereignRiskPredictor, self).__init__()
        
        # Economic indicators branch
        self.economic_mlp = nn.Sequential(
            nn.Linear(num_economic_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)]
        )
        
        # Network features branch
        self.network_mlp = nn.Sequential(
            nn.Linear(num_network_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
               economic_features: torch.Tensor,
               network_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict sovereign risk probability
        
        Args:
            economic_features: Economic indicators [batch_size, num_economic_features]
            network_features: Network-based risk features [batch_size, num_network_features]
            
        Returns:
            Dict containing risk predictions and intermediate features
        """
        # Process economic indicators
        economic_embedding = self.economic_mlp(economic_features)
        
        # Process network features
        network_embedding = self.network_mlp(network_features)
        
        # Combine features
        combined = torch.cat([economic_embedding, network_embedding], dim=1)
        
        # Predict crisis probability
        crisis_prob = self.prediction_head(combined)
        
        return {
            'crisis_probability': crisis_prob,
            'economic_embedding': economic_embedding,
            'network_embedding': network_embedding
        }
