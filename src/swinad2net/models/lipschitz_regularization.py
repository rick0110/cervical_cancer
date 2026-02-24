"""Lipschitz Regularization Module.

This module provides utilities to compute and regularize the Lipschitz constant
of neural network layers. The Lipschitz constant bounds how much the output can
change relative to input changes, which is crucial for:
- Training stability
- Generalization
- Robustness to adversarial examples

The regularization includes:
- Penalty for large Lipschitz constants (prevents exploding gradients)
- Penalty for small Lipschitz constants (prevents vanishing gradients/dead neurons)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


def estimate_spectral_norm(weight: torch.Tensor, n_power_iterations: int = 1) -> torch.Tensor:
    """
    Estimate the spectral norm (largest singular value) of a weight matrix
    using power iteration method.
    
    Args:
        weight: Weight tensor (can be 2D, 3D or 4D for conv layers)
        n_power_iterations: Number of power iterations for estimation
        
    Returns:
        Estimated spectral norm (scalar tensor)
    """
    # Reshape weight to 2D matrix for spectral norm computation
    if weight.dim() > 2:
        # For conv layers: reshape (out_ch, in_ch, kH, kW) -> (out_ch, in_ch * kH * kW)
        weight_mat = weight.reshape(weight.size(0), -1)
    else:
        weight_mat = weight
    
    # Initialize random vectors for power iteration
    h, w = weight_mat.shape
    u = torch.randn(h, device=weight.device, dtype=weight.dtype)
    u = u / u.norm()
    
    with torch.no_grad():
        for _ in range(n_power_iterations):
            # v = W^T u / ||W^T u||
            v = torch.mv(weight_mat.t(), u)
            v = v / (v.norm() + 1e-12)
            # u = W v / ||W v||
            u = torch.mv(weight_mat, v)
            u = u / (u.norm() + 1e-12)
    
    # sigma = u^T W v
    sigma = torch.dot(u, torch.mv(weight_mat, v))
    
    return sigma.abs()


def compute_exact_spectral_norm(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the exact spectral norm using SVD.
    More accurate but slower than power iteration.
    
    Args:
        weight: Weight tensor
        
    Returns:
        Exact spectral norm (largest singular value)
    """
    if weight.dim() > 2:
        weight_mat = weight.reshape(weight.size(0), -1)
    else:
        weight_mat = weight
    
    # Use SVD to get exact spectral norm
    try:
        singular_values = torch.linalg.svdvals(weight_mat)
        return singular_values[0]
    except:
        # Fallback to power iteration if SVD fails
        return estimate_spectral_norm(weight, n_power_iterations=10)


class LipschitzRegularizer:
    """
    Computes Lipschitz regularization loss for neural networks.
    
    The regularization penalizes:
    1. Large Lipschitz constants (for stability)
    2. Small Lipschitz constants (for expressiveness)
    
    The total Lipschitz constant of the network is bounded by the product
    of individual layer Lipschitz constants.
    """
    
    def __init__(self,
                 upper_bound: float = 10.0,
                 lower_bound: float = 0.1,
                 lambda_upper: float = 0.01,
                 lambda_lower: float = 0.001,
                 use_exact_svd: bool = False,
                 n_power_iterations: int = 1,
                 layer_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the Lipschitz regularizer.
        
        Args:
            upper_bound: Target upper bound for layer Lipschitz constants
            lower_bound: Target lower bound for layer Lipschitz constants
            lambda_upper: Weight for upper bound penalty
            lambda_lower: Weight for lower bound penalty
            use_exact_svd: If True, use exact SVD (slower but more accurate)
            n_power_iterations: Number of power iterations for spectral norm estimation
            layer_weights: Optional dict mapping layer names to custom weights
        """
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.lambda_upper = lambda_upper
        self.lambda_lower = lambda_lower
        self.use_exact_svd = use_exact_svd
        self.n_power_iterations = n_power_iterations
        self.layer_weights = layer_weights or {}
        
    def compute_layer_lipschitz(self, module: nn.Module, name: str = "") -> Dict[str, torch.Tensor]:
        """
        Compute Lipschitz constants for all applicable layers in a module.
        
        Args:
            module: PyTorch module to analyze
            name: Base name for layer identification
            
        Returns:
            Dictionary mapping layer names to their Lipschitz constants
        """
        lipschitz_constants = {}
        
        for child_name, child in module.named_modules():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Handle different layer types
            if isinstance(child, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                if hasattr(child, 'weight') and child.weight is not None:
                    if self.use_exact_svd:
                        sigma = compute_exact_spectral_norm(child.weight)
                    else:
                        sigma = estimate_spectral_norm(child.weight, self.n_power_iterations)
                    lipschitz_constants[full_name] = sigma
                    
            elif isinstance(child, nn.LayerNorm):
                # LayerNorm has Lipschitz constant approximately 
                # bounded by gamma / (std + eps) which is typically ~1
                if hasattr(child, 'weight') and child.weight is not None:
                    gamma = child.weight.abs().max()
                    lipschitz_constants[full_name] = gamma
                    
            elif isinstance(child, nn.BatchNorm2d):
                # BatchNorm Lipschitz ~ gamma / sqrt(var + eps)
                if hasattr(child, 'weight') and child.weight is not None:
                    gamma = child.weight.abs()
                    var = child.running_var if child.running_var is not None else torch.ones_like(gamma)
                    lip = (gamma / torch.sqrt(var + child.eps)).max()
                    lipschitz_constants[full_name] = lip
                    
        return lipschitz_constants
    
    def compute_regularization_loss(self, 
                                     model: nn.Module,
                                     return_details: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute the total Lipschitz regularization loss.
        
        The loss penalizes:
        - Layers with Lipschitz > upper_bound: lambda_upper * (L - upper_bound)^2
        - Layers with Lipschitz < lower_bound: lambda_lower * (lower_bound - L)^2
        
        Args:
            model: Neural network model
            return_details: If True, return per-layer details
            
        Returns:
            Tuple of (total_loss, optional_details_dict)
        """
        lipschitz_constants = self.compute_layer_lipschitz(model)
        
        upper_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        lower_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        
        details = {} if return_details else None
        
        for name, lip_const in lipschitz_constants.items():
            # Get layer-specific weight if defined
            weight = self.layer_weights.get(name, 1.0)
            
            # Upper bound penalty (stability)
            if lip_const > self.upper_bound:
                penalty = weight * (lip_const - self.upper_bound) ** 2
                upper_penalty = upper_penalty + penalty
                
            # Lower bound penalty (expressiveness)
            if lip_const < self.lower_bound:
                penalty = weight * (self.lower_bound - lip_const) ** 2
                lower_penalty = lower_penalty + penalty
            
            if return_details:
                details[name] = {
                    'lipschitz': lip_const.item(),
                    'in_range': self.lower_bound <= lip_const.item() <= self.upper_bound
                }
        
        total_loss = self.lambda_upper * upper_penalty + self.lambda_lower * lower_penalty
        
        if return_details:
            details['_summary'] = {
                'total_loss': total_loss.item(),
                'upper_penalty': upper_penalty.item(),
                'lower_penalty': lower_penalty.item(),
                'num_layers': len(lipschitz_constants),
                'product_lipschitz': np.prod([v.item() for v in lipschitz_constants.values()]) if lipschitz_constants else 1.0
            }
        
        return total_loss, details
    
    def get_network_lipschitz_bound(self, model: nn.Module) -> float:
        """
        Estimate the upper bound on the network's Lipschitz constant.
        
        For a feedforward network, the Lipschitz constant is bounded by
        the product of individual layer Lipschitz constants.
        
        Args:
            model: Neural network model
            
        Returns:
            Estimated Lipschitz bound for the entire network
        """
        lipschitz_constants = self.compute_layer_lipschitz(model)
        
        if not lipschitz_constants:
            return 1.0
            
        # Network Lipschitz <= product of layer Lipschitz constants
        product = 1.0
        for lip_const in lipschitz_constants.values():
            product *= lip_const.item()
            
        return product


class SpectralNormConstraint(nn.Module):
    """
    A module wrapper that applies spectral normalization to constrain
    the Lipschitz constant of a layer to be at most `max_norm`.
    """
    
    def __init__(self, module: nn.Module, max_norm: float = 1.0, n_power_iterations: int = 1):
        """
        Args:
            module: The module to wrap (e.g., nn.Linear, nn.Conv2d)
            max_norm: Maximum spectral norm allowed
            n_power_iterations: Power iterations for estimation
        """
        super().__init__()
        self.module = module
        self.max_norm = max_norm
        self.n_power_iterations = n_power_iterations
        
        # Register buffers for power iteration
        if hasattr(module, 'weight'):
            weight = module.weight
            h, w = weight.reshape(weight.size(0), -1).shape
            self.register_buffer('_u', torch.randn(h))
            self.register_buffer('_v', torch.randn(w))
            
    def _update_uv(self, weight_mat: torch.Tensor):
        """Update u and v vectors using power iteration."""
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                self._v = torch.mv(weight_mat.t(), self._u)
                self._v = self._v / (self._v.norm() + 1e-12)
                self._u = torch.mv(weight_mat, self._v)
                self._u = self._u / (self._u.norm() + 1e-12)
                
    def forward(self, x):
        if hasattr(self.module, 'weight') and self.module.weight is not None:
            weight = self.module.weight
            weight_mat = weight.reshape(weight.size(0), -1)
            
            self._update_uv(weight_mat)
            
            # Compute spectral norm
            sigma = torch.dot(self._u, torch.mv(weight_mat, self._v))
            
            # Scale weight if spectral norm exceeds max_norm
            if sigma > self.max_norm:
                scale = self.max_norm / sigma
                # Temporarily scale weights
                with torch.no_grad():
                    self.module.weight.data = weight * scale
                    
        return self.module(x)


def apply_lipschitz_constraint(model: nn.Module,
                                max_lipschitz: float = 1.0,
                                layers_to_constrain: Optional[List[str]] = None) -> nn.Module:
    """
    Apply Lipschitz constraint to specified layers in a model.
    
    Args:
        model: The model to modify
        max_lipschitz: Maximum Lipschitz constant per layer
        layers_to_constrain: List of layer names to constrain (None = all applicable)
        
    Returns:
        Modified model with Lipschitz constraints
    """
    for name, module in model.named_modules():
        if layers_to_constrain is not None and name not in layers_to_constrain:
            continue
            
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Apply spectral normalization
            nn.utils.spectral_norm(module, n_power_iterations=1)
            
    return model


# Utility functions for training integration
def lipschitz_loss_wrapper(model: nn.Module,
                            criterion: nn.Module,
                            regularizer: LipschitzRegularizer):
    """
    Create a loss function that includes Lipschitz regularization.
    
    Args:
        model: The model being trained
        criterion: Base loss function (e.g., CrossEntropyLoss)
        regularizer: LipschitzRegularizer instance
        
    Returns:
        Wrapped loss function
    """
    def loss_fn(outputs, targets):
        base_loss = criterion(outputs, targets)
        lip_loss, _ = regularizer.compute_regularization_loss(model)
        return base_loss + lip_loss
    
    return loss_fn
