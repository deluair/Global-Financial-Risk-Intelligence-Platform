"""
Financial Network Analysis Module for GFRIP
Handles construction and analysis of financial networks for systemic risk assessment
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from ..models.risk_models import FinancialContagionGNN
import logging

logger = logging.getLogger(__name__)

class FinancialNetworkBuilder:
    """
    Constructs and analyzes financial networks from various data sources
    """
    
    def __init__(self, 
                 directed: bool = True, 
                 self_loops: bool = False):
        """
        Initialize the financial network builder
        
        Args:
            directed: Whether the network is directed
            self_loops: Whether to include self-loops in the network
        """
        self.directed = directed
        self.self_loops = self_loops
        self.network = None
        
    def construct_network(self, 
                          institutions_data: pd.DataFrame,
                          exposures_matrix: Union[np.ndarray, pd.DataFrame],
                          time_window: Optional[Tuple[str, str]] = None) -> nx.Graph:
        """
        Construct a financial network from institutions and exposures data
        
        Args:
            institutions_data: DataFrame containing institution metadata
            exposures_matrix: Matrix of financial exposures between institutions
            time_window: Optional time window for temporal analysis (start_date, end_date)
            
        Returns:
            NetworkX graph representing the financial network
        """
        logger.info(f"Constructing financial network with {len(institutions_data)} institutions")
        
        # Create graph
        if self.directed:
            self.network = nx.DiGraph()
        else:
            self.network = nx.Graph()
            
        # Add nodes with attributes
        for _, row in institutions_data.iterrows():
            self.network.add_node(
                row['institution_id'],
                **row.to_dict()
            )
            
        # Add edges with exposure weights
        if isinstance(exposures_matrix, pd.DataFrame):
            exposures_matrix = exposures_matrix.values
            
        n = len(institutions_data)
        if exposures_matrix.shape != (n, n):
            raise ValueError(f"Exposures matrix must be square with size {n}x{n}")
            
        for i in range(n):
            for j in range(n):
                if i == j and not self.self_loops:
                    continue
                    
                exposure = exposures_matrix[i, j]
                if exposure > 0:  # Only add edges with positive exposure
                    self.network.add_edge(
                        institutions_data.iloc[i]['institution_id'],
                        institutions_data.iloc[j]['institution_id'],
                        weight=exposure
                    )
                    
        logger.info(f"Network constructed with {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges")
        return self.network
    
    def calculate_network_metrics(self) -> Dict[str, Any]:
        """
        Calculate key network metrics for systemic risk assessment
        
        Returns:
            Dictionary of network metrics
        """
        if self.network is None:
            raise ValueError("No network has been constructed. Call construct_network() first.")
            
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = self.network.number_of_nodes()
        metrics['num_edges'] = self.network.number_of_edges()
        metrics['density'] = nx.density(self.network)
        
        # Centrality measures
        metrics['degree_centrality'] = nx.degree_centrality(self.network)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(self.network, weight='weight')
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(self.network, weight='weight')
        
        # Systemic importance
        metrics['pagerank'] = nx.pagerank(self.network, weight='weight')
        
        # Connectivity
        if self.directed:
            metrics['is_strongly_connected'] = nx.is_strongly_connected(self.network)
            metrics['strongly_connected_components'] = list(nx.strongly_connected_components(self.network))
        else:
            metrics['is_connected'] = nx.is_connected(self.network)
            metrics['connected_components'] = list(nx.connected_components(self.network))
        
        return metrics
    
    def identify_systemically_important_nodes(self, 
                                            top_n: int = 10,
                                            metric: str = 'pagerank') -> List[Tuple[Any, float]]:
        """
        Identify systemically important nodes using the specified centrality metric
        
        Args:
            top_n: Number of top nodes to return
            metric: Centrality metric to use ('pagerank', 'betweenness', 'eigenvector', 'degree')
            
        Returns:
            List of (node_id, score) tuples, sorted by score in descending order
        """
        if self.network is None:
            raise ValueError("No network has been constructed. Call construct_network() first.")
            
        # Calculate appropriate centrality measure
        if metric == 'pagerank':
            scores = nx.pagerank(self.network, weight='weight')
        elif metric == 'betweenness':
            scores = nx.betweenness_centrality(self.network, weight='weight')
        elif metric == 'eigenvector':
            scores = nx.eigenvector_centrality_numpy(self.network, weight='weight')
        elif metric == 'degree':
            scores = dict(self.network.degree(weight='weight'))
        else:
            raise ValueError(f"Unsupported centrality metric: {metric}")
            
        # Sort nodes by score
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:top_n]


class ContagionRiskAnalyzer:
    """
    Advanced contagion risk analysis using graph neural networks
    """
    
    def __init__(self, 
                 node_features: int = 128,
                 edge_features: int = 64,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the contagion risk analyzer
        
        Args:
            node_features: Number of input node features
            edge_features: Number of input edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = FinancialContagionGNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        self.network_builder = FinancialNetworkBuilder()
        
    def prepare_graph_data(self, 
                         institutions_data: pd.DataFrame,
                         exposures_matrix: Union[np.ndarray, pd.DataFrame]) -> Data:
        """
        Prepare graph data for GNN input
        
        Args:
            institutions_data: DataFrame containing institution features
            exposures_matrix: Matrix of financial exposures
            
        Returns:
            PyTorch Geometric Data object
        """
        # Build network
        network = self.network_builder.construct_network(institutions_data, exposures_matrix)
        
        # Convert to PyTorch Geometric format
        node_features = torch.tensor(
            institutions_data.drop('institution_id', axis=1).values,
            dtype=torch.float
        )
        
        # Get edge indices and attributes
        edge_index = []
        edge_attr = []
        
        for i, j, data in network.edges(data=True):
            edge_index.append([i, j])
            edge_attr.append(data.get('weight', 1.0))
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
    
    def analyze_contagion_risk(self, 
                             institutions_data: pd.DataFrame,
                             exposures_matrix: Union[np.ndarray, pd.DataFrame],
                             shock_scenario: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze contagion risk in the financial network
        
        Args:
            institutions_data: DataFrame containing institution features
            exposures_matrix: Matrix of financial exposures
            shock_scenario: Optional shock scenario to apply
            
        Returns:
            Dictionary containing risk analysis results
        """
        # Prepare graph data
        graph_data = self.prepare_graph_data(institutions_data, exposures_matrix)
        graph_data = graph_data.to(self.device)
        
        # Apply shock scenario if provided
        if shock_scenario:
            self._apply_shock(graph_data, shock_scenario)
        
        # Run GNN model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr
            )
        
        # Process outputs
        contagion_risk = outputs['contagion_risk'].cpu().numpy().flatten()
        systemic_risk = outputs['systemic_risk'].item()
        
        # Calculate network metrics
        network_metrics = self.network_builder.calculate_network_metrics()
        
        # Identify systemically important nodes
        top_risky_nodes = self.network_builder.identify_systemically_important_nodes()
        
        return {
            'contagion_risk_scores': dict(zip(institutions_data['institution_id'], contagion_risk)),
            'systemic_risk_score': systemic_risk,
            'network_metrics': network_metrics,
            'systemically_important_nodes': top_risky_nodes,
            'node_embeddings': outputs['node_embeddings'].cpu().numpy()
        }
    
    def _apply_shock(self, graph_data: Data, shock_scenario: Dict) -> None:
        """
        Apply a shock scenario to the graph data
        
        Args:
            graph_data: PyTorch Geometric Data object
            shock_scenario: Dictionary defining the shock scenario
        """
        if 'node_attribute' in shock_scenario:
            # Apply shock to node features
            attr_name = shock_scenario['node_attribute']
            node_indices = shock_scenario.get('node_indices', slice(None))
            shock_value = shock_scenario['value']
            
            if isinstance(node_indices, list):
                node_indices = torch.tensor(node_indices, device=graph_data.x.device)
            
            graph_data.x[node_indices, graph_data.x_names.index(attr_name)] *= (1 + shock_value)
            
        elif 'edge_attribute' in shock_scenario:
            # Apply shock to edge features
            attr_name = shock_scenario['edge_attribute']
            edge_indices = shock_scenario.get('edge_indices', slice(None))
            shock_value = shock_scenario['value']
            
            if isinstance(edge_indices, list):
                edge_indices = torch.tensor(edge_indices, device=graph_data.edge_attr.device)
            
            graph_data.edge_attr[edge_indices, graph_data.edge_attr_names.index(attr_name)] *= (1 + shock_value)
        
        # Add shock metadata to graph
        if not hasattr(graph_data, 'shock_history'):
            graph_data.shock_history = []
        graph_data.shock_history.append(shock_scenario)
    
    def stress_test(self, 
                   institutions_data: pd.DataFrame,
                   exposures_matrix: Union[np.ndarray, pd.DataFrame],
                   shock_scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Run stress tests with multiple shock scenarios
        
        Args:
            institutions_data: DataFrame containing institution features
            exposures_matrix: Matrix of financial exposures
            shock_scenarios: List of shock scenarios to apply
            
        Returns:
            Dictionary containing stress test results
        """
        results = {}
        
        # Baseline analysis (no shocks)
        baseline = self.analyze_contagion_risk(institutions_data, exposures_matrix)
        results['baseline'] = baseline
        
        # Apply each shock scenario
        for i, scenario in enumerate(shock_scenarios):
            scenario_name = scenario.get('name', f'scenario_{i+1}')
            logger.info(f"Running stress test scenario: {scenario_name}")
            
            # Copy the baseline data and apply shock
            shocked_data = self.prepare_graph_data(institutions_data, exposures_matrix)
            self._apply_shock(shocked_data, scenario)
            
            # Analyze with shock
            scenario_result = self.analyze_contagion_risk(
                institutions_data,
                exposures_matrix,
                scenario
            )
            
            # Calculate impact metrics
            baseline_risk = baseline['systemic_risk_score']
            shocked_risk = scenario_result['systemic_risk_score']
            risk_change = (shocked_risk - baseline_risk) / baseline_risk if baseline_risk > 0 else 0
            
            results[scenario_name] = {
                **scenario_result,
                'risk_change': risk_change,
                'shock_impact': scenario
            }
        
        return results
