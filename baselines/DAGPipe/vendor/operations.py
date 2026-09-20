"""
Differentiable Preprocessing Operations for DiffDAG-Prep

This module implements differentiable wrappers for common preprocessing operations,
enabling gradient-based optimization of preprocessing pipelines.

Operations are categorized into four types:
1. Imputation: Handle missing values in data
2. Scaling: Normalize feature scales for better model convergence
3. Encoding: Convert categorical features to numerical representations
4. Transformation: Create non-linear transformations of features

All operations are differentiable, allowing them to be optimized via gradient descent
as part of an end-to-end learning pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod


class DifferentiableOperation(nn.Module, ABC):
    """Base class for differentiable preprocessing operations."""
    
    def __init__(self, name: str, op_type: str):
        super().__init__()
        self.name = name
        self.op_type = op_type  # 'imputer', 'scaler', 'encoder', 'transformer'
    
    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply the operation.
        
        Args:
            x: Input tensor [batch_size, num_features] or [batch_size, 1]
            mask: Optional missing value mask [batch_size, num_features]
        
        Returns:
            Transformed tensor
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, type={self.op_type})"



# =============================================================================
# IMPUTATION OPERATIONS
# =============================================================================

class MeanImputer(DifferentiableOperation):
    """
    Mean Imputation: Replace missing values with the column mean.

    This is a simple and commonly used imputation strategy. Missing values are
    replaced with the mean of non-missing values in each feature column.

    Use cases:
    - Data is approximately normally distributed
    - Missing values are randomly distributed (MCAR - Missing Completely At Random)
    - Quick baseline imputation strategy

    Limitations:
    - Can be affected by outliers
    - May not preserve data distribution
    - Ignores correlations between features
    """
    
    def __init__(self):
        super().__init__(name='mean_imputer', op_type='imputer')
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            # mask = torch.isnan(x).float()
            mask = ((x + 99999).abs() < 1e-3).float()

        # Compute mean ignoring NaN
        x_filled = torch.where(((x + 99999).abs() < 1e-3), torch.zeros_like(x), x)
        valid_count = (1 - mask).sum(dim=0, keepdim=True).clamp(min=1)
        mean_val = x_filled.sum(dim=0, keepdim=True) / valid_count

        # Fill missing values
        output = torch.where(mask.bool(), mean_val.expand_as(x), x_filled)
        return output


class MedianImputer(DifferentiableOperation):
    """
    Median Imputation: Replace missing values with the column median.

    Median is more robust to outliers compared to mean. It represents the
    middle value when data is sorted, making it suitable for skewed distributions.

    Use cases:
    - Data contains outliers that would affect the mean
    - Feature distribution is skewed
    - More robust baseline than mean imputation

    Limitations:
    - Still ignores feature correlations
    - For small datasets, may not be representative
    - Less computationally efficient than mean imputation
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Smoothing parameter for soft median approximation.
                        Lower values make it closer to true median.
        """
        super().__init__(name='median_imputer', op_type='imputer')
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            # mask = torch.isnan(x).float()
            mask = ((x + 99999).abs() < 1e-3).float()
        
        x_filled = torch.where(((x + 99999).abs() < 1e-3), torch.zeros_like(x), x)
        
        # Soft median using weighted average around median position
        sorted_x, _ = torch.sort(x_filled, dim=0)
        n = x.shape[0]
        positions = torch.arange(n, device=x.device, dtype=x.dtype).unsqueeze(1)
        median_pos = (n - 1) / 2.0
        weights = F.softmax(-torch.abs(positions - median_pos) / self.temperature, dim=0)
        median_val = (weights * sorted_x).sum(dim=0, keepdim=True)
        
        output = torch.where(mask.bool(), median_val.expand_as(x), x_filled)
        return output


class KNNImputer(DifferentiableOperation):
    """
    K-Nearest Neighbors Imputation: Fill missing values using similar samples.

    For each sample with missing values, this method finds the k most similar
    samples (based on Euclidean distance) and uses a weighted average of their
    values to fill in the missing entries.

    Use cases:
    - Data has local structure or clusters
    - Similar samples should have similar feature values
    - Missing values depend on other features (MAR - Missing At Random)
    - Want to preserve feature correlations

    Advantages:
    - Captures local data structure
    - Preserves correlations between features
    - More sophisticated than simple mean/median imputation

    Limitations:
    - Computationally expensive for large datasets
    - Sensitive to distance metric choice
    - Performance depends on appropriate k value
    - Requires data normalization for meaningful distances
    """
    
    def __init__(self, k: int = 5, temperature: float = 1.0):
        """
        Args:
            k: Number of nearest neighbors to consider. Typical values: 3-10.
            temperature: Softmax temperature for weighting neighbors.
                        Lower values give more weight to closer neighbors.
        """
        super().__init__(name='knn_imputer', op_type='imputer')
        self.k = k
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            # mask = torch.isnan(x).float()
            mask = ((x + 99999).abs() < 1e-3).float()
        
        x_filled = torch.where(((x + 99999).abs() < 1e-3), torch.zeros_like(x), x)
        
        batch_size, num_features = x.shape
        # x_filled = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Compute pairwise distances
        x_norm = x_filled / (x_filled.std(dim=0, keepdim=True) + 1e-8)
        dist = torch.cdist(x_norm, x_norm)  # [batch, batch]
        
        # Mask self-distances
        dist = dist + torch.eye(batch_size, device=x.device) * 1e9
        
        # Soft top-k using softmax
        weights = F.softmax(-dist / self.temperature, dim=1)  # [batch, batch]
        
        # Compute KNN imputed values
        knn_values = torch.matmul(weights, x_filled)  # [batch, features]
        
        output = torch.where(mask.bool(), knn_values, x_filled)
        return output


class ZeroImputer(DifferentiableOperation):
    """
    Zero Imputation: Replace missing values with zero.

    Simple imputation strategy that sets all missing values to zero.
    This is commonly used in sparse data scenarios or when zero is a
    meaningful "missing" value in the domain.

    Use cases:
    - Sparse features where zero indicates absence
    - Count data with missing counts
    - Neural networks where zero is a neutral value
    - Quick baseline for sparse datasets

    Advantages:
    - Very fast and simple
    - Preserves sparsity of data
    - Works well with sparse matrix operations
    - No assumptions about data distribution

    Limitations:
    - Can introduce bias if zero is not appropriate
    - Ignores all information about missingness
    - May not work well for features that don't have zero as natural value
    """


class LearnedImputer(DifferentiableOperation):
    """
    Learned Constant Imputation: Learn optimal fill values via backpropagation.

    Instead of using a heuristic like mean or median, this operation learns
    the optimal fill values for each feature during model training. The fill
    values are trainable parameters that are optimized together with the model.

    Use cases:
    - Want data-driven imputation values
    - Domain knowledge suggests specific fill values
    - Feature distributions are complex
    - End-to-end learning pipeline

    Advantages:
    - Learned values are optimized for downstream task
    - Flexible and data-adaptive
    - Can capture domain-specific patterns
    - Differentiable and trainable

    Limitations:
    - Adds trainable parameters (increased memory)
    - May overfit if dataset is small
    - Requires sufficient training data
    - Less interpretable than mean/median
    """
    
    def __init__(self, num_features: int):
        """
        Args:
            num_features: Number of features in the input data.
        """
        super().__init__(name='learned_imputer', op_type='imputer')
        self.fill_values = nn.Parameter(torch.zeros(1, num_features))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.isnan(x).float()
        
        x_filled = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        output = torch.where(mask.bool(), self.fill_values.expand_as(x), x_filled)
        return output


class MissingIndicator(DifferentiableOperation):
    """Add missing indicator features."""
    
    def __init__(self):
        super().__init__(name='missing_indicator', op_type='imputer')
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.isnan(x).float()
        
        # Just return the mask as indicator
        # Note: In actual use, this would concatenate with imputed values
        x_filled = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x_filled


# =============================================================================
# SCALING OPERATIONS
# =============================================================================

class StandardScaler(DifferentiableOperation):
    """
    Standard Scaling (Z-score Normalization): Scale to mean=0, std=1.

    Transforms features to have zero mean and unit standard deviation.
    Formula: (x - mean) / std

    Use cases:
    - Features have different scales or units
    - Data approximately follows normal distribution
    - Most common default preprocessing for many ML models
    - Neural networks, SVMs, k-NN typically benefit

    Advantages:
    - Standardizes feature scales
    - Helps gradient descent converge faster
    - Reduces feature bias in distance-based models
    - Simple and interpretable

    Limitations:
    - Sensitive to outliers (they affect mean and std)
    - May not be ideal for skewed distributions
    - Loses interpretability of original scale
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(name='standard_scaler', op_type='scaler')
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + self.eps
        return (x - mean) / std


class RobustScaler(DifferentiableOperation):
    """
    Robust Scaling: Scale using median and IQR (Interquartile Range).

    Uses median and IQR instead of mean and std, making it more robust
    to outliers. Formula: (x - median) / IQR

    Use cases:
    - Data contains significant outliers
    - Feature distribution is skewed
    - Want to preserve outlier information without letting them dominate
    - Heavy-tailed distributions

    Advantages:
    - Robust to outliers
    - Better for skewed distributions
    - Median is a better measure of central tendency for non-Gaussian data
    - IQR is robust to extreme values

    Limitations:
    - Less interpretable than standard scaling
    - May not preserve Gaussian-like distributions
    - More computationally expensive (quantile computation)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(name='robust_scaler', op_type='scaler')
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        median = x.median(dim=0, keepdim=True).values
        q75 = torch.quantile(x, 0.75, dim=0, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=0, keepdim=True)
        iqr = q75 - q25 + self.eps
        return (x - median) / iqr


class MinMaxScaler(DifferentiableOperation):
    """
    Min-Max Scaling: Scale features to [0, 1] range.

    Linearly transforms features to have a minimum of 0 and maximum of 1.
    Formula: (x - min) / (max - min)

    Use cases:
    - Features have bounded ranges
    - Want to preserve the original distribution shape
    - Models sensitive to feature magnitude (neural networks)
    - Image data, distances, percentages

    Advantages:
    - Preserves zero values and sparsity
    - Simple and interpretable
    - Bounded output range [0, 1]
    - Good for algorithms using distance or similarity

    Limitations:
    - Very sensitive to outliers
    - A single outlier can compress all other values
    - Does not handle new values outside training range
    - Loses information about variance
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(name='minmax_scaler', op_type='scaler')
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        min_val = x.min(dim=0, keepdim=True).values
        max_val = x.max(dim=0, keepdim=True).values
        range_val = max_val - min_val + self.eps
        return (x - min_val) / range_val



class MaxAbsScaler(DifferentiableOperation):
    """
    MaxAbs Scaling: Scale features to [-1, 1] range.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__(name='maxabs_scaler', op_type='scaler')
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        max_val = torch.abs(x).max(dim=0, keepdim=True).values
        return x / (max_val + self.eps)


class QuantileScaler(DifferentiableOperation):
    """
    Quantile Transformation: Map to uniform distribution via quantiles.

    Transforms features to follow a uniform distribution by mapping
    values to their rank percentiles. This makes different features
    more comparable regardless of their original distributions.

    Use cases:
    - Features have very different distributions
    - Want to make features follow similar distribution
    - Non-linear transformations are acceptable
    - Data is not well-modeled by simple scaling

    Advantages:
    - Makes different distributions comparable
    - Robust to outliers
    - Handles skewed and multi-modal distributions
    - Can reveal non-linear relationships

    Limitations:
    - Loses original distribution shape information
    - More complex and computationally expensive
    - May not preserve monotonic relationships
    - Requires sufficient data for accurate quantiles
    """

    def __init__(self, num_quantiles: int = 100, eps: float = 1e-8):
        """
        Args:
            num_quantiles: Number of quantile bins (for approximate transform).
            eps: Small epsilon for numerical stability.
        """
        super().__init__(name='quantile_scaler', op_type='scaler')
        self.num_quantiles = num_quantiles
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Approximate quantile transform using empirical CDF
        batch_size, num_features = x.shape

        # Sort and compute ranks
        sorted_x, sort_idx = torch.sort(x, dim=0)
        ranks = torch.argsort(sort_idx, dim=0).float()

        # Normalize to [0, 1]
        normalized_ranks = ranks / (batch_size - 1 + self.eps)

        return normalized_ranks


class LogScaler(DifferentiableOperation):
    """
    Log Transformation: Apply log for right-skewed data.

    Transforms features using logarithm to reduce the impact of
    large values and make right-skewed distributions more symmetric.
    Formula: log(x - min + 1)

    Use cases:
    - Data follows exponential or power-law distribution
    - Features span multiple orders of magnitude
    - Right-skewed distributions (income, population, prices)
    - Want to reduce heteroscedasticity

    Advantages:
    - Compresses large values more than small values
    - Makes right-skewed distributions more symmetric
    - Multiplicative relationships become additive
    - Can stabilize variance

    Limitations:
    - Only works with positive values (needs shift)
    - Cannot handle zeros directly
    - May not be appropriate for all data
    - Interpretation becomes more complex
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(name='log_scaler', op_type='scaler')
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Shift to ensure positive values
        min_val = x.min(dim=0, keepdim=True).values
        x_shifted = x - min_val + 1
        return torch.log(x_shifted + self.eps)


class IdentityScaler(DifferentiableOperation):
    """
    Identity Operation: No transformation (pass-through).

    Returns input unchanged. Useful as a baseline operation or when
    no preprocessing is needed for certain features.

    Use cases:
    - Feature is already in optimal form
    - Baseline for comparison
    - Conditional feature preprocessing
    - Some features should not be transformed

    Advantages:
    - No change to data
    - Computationally efficient
    - Preserves all information
    - Can be used as fallback

    Limitations:
    - Doesn't address any data issues
    - May not help model convergence
    - Features remain on original scales
    """
    def __init__(self):
        super().__init__(name='identity', op_type='scaler')

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


# =============================================================================
# ENCODING OPERATIONS
# =============================================================================

class TargetEncoder(DifferentiableOperation):
    """
    Target Encoding: Encode categorical values by their target mean.

    For each categorical value, compute mean of target variable
    for samples with that value. This captures the relationship between
    categorical feature and target variable.

    Formula: encoded = (count * local_mean + smoothing * global_mean) / (count + smoothing)

    Use cases:
    - High-cardinality categorical features (zip codes, user IDs)
    - Categorical feature has predictive relationship with target
    - Tree-based models or linear models
    - Want to capture target-related information

    Advantages:
    - Captures target information directly
    - Works well for high-cardinality categories
    - Fewer dimensions than one-hot encoding
    - Smoothing prevents overfitting for rare categories

    Limitations:
    - Requires target variable (supervised)
    - Can cause overfitting if not properly smoothed
    - Data leakage risk if not carefully implemented (need proper CV)
    - Not suitable for online learning without retraining
    """

    def __init__(self, smoothing: float = 1.0):
        """
        Args:
            smoothing: Smoothing factor to prevent overfitting for rare categories.
                      Higher values give more weight to global mean.
        """
        super().__init__(name='target_encoder', op_type='encoder')
        self.smoothing = smoothing
        self.global_mean = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        if target is None:
            return x  # Cannot encode without target

        # Simple smoothed target encoding
        # For each unique value, compute smoothed mean of target
        unique_vals = torch.unique(x)
        encoded = torch.zeros_like(x)

        global_mean = target.mean()

        for val in unique_vals:
            mask_val = (x == val)
            count = mask_val.sum()
            if count > 0:
                local_mean = target[mask_val.squeeze()].mean()
                # Smoothed encoding
                smoothed = (count * local_mean + self.smoothing * global_mean) / (count + self.smoothing)
                encoded = torch.where(mask_val, smoothed, encoded)

        return encoded


class FrequencyEncoder(DifferentiableOperation):
    """
    Frequency Encoding: Encode categorical values by their occurrence frequency.

    Replace each categorical value with its frequency (count/total) in
    the dataset. This captures how common each category is.

    Use cases:
    - Rare/common categories have different meanings
    - High-cardinality categorical features
    - Tree-based models
    - Simple baseline encoding

    Advantages:
    - Simple and fast
    - Works for high-cardinality features
    - No target leakage (unsupervised)
    - Captures importance via frequency

    Limitations:
    - Loses actual categorical information
    - Different categories with same frequency get same encoding
    - Doesn't capture relationship with target
    - May not be optimal if frequency isn't meaningful
    """

    def __init__(self):
        super().__init__(name='frequency_encoder', op_type='encoder')

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        unique_vals, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
        frequencies = counts[inverse].float() / batch_size
        return frequencies.view_as(x)


# =============================================================================
# TRANSFORM OPERATIONS
# =============================================================================

class PolynomialTransform(DifferentiableOperation):
    """
    Polynomial Transformation: Generate polynomial features.

    Creates non-linear feature transformations using polynomial functions.
    Current implementation generates x^2 (degree=2). A full implementation
    would generate all polynomial combinations up to the specified degree.

    Use cases:
    - Capturing non-linear relationships
    - Linear models that need non-linear features
    - Feature interactions are important
    - Simple baseline for feature engineering

    Advantages:
    - Captures non-linear patterns
    - Can improve model expressiveness
    - Simple to implement
    - Works well with linear models

    Limitations:
    - Can create many features (combinatorial explosion)
    - May lead to overfitting with high degree
    - Sensitive to outliers (polynomials amplify extreme values)
    - Current simplified version only does x^2
    """

    def __init__(self, degree: int = 2):
        """
        Args:
            degree: Polynomial degree (currently only degree=2 is fully implemented).
        """
        super().__init__(name='polynomial', op_type='transformer')
        self.degree = degree

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Simple polynomial: return x^2 for degree=2
        # In full implementation, would generate all polynomial combinations
        if self.degree == 2:
            return x ** 2
        return x


class DiscretizeTransform(DifferentiableOperation):
    """
    Discretization: Convert continuous features to bins.

    Bins continuous features into discrete categories using soft binning
    (differentiable via softmax). This can help models capture
    non-linear relationships and handle outliers.

    Use cases:
    - Want to reduce sensitivity to outliers
    - Non-linear relationships are step-like
    - Feature engineering for tree-based models
    - Simplifying complex continuous patterns

    Advantages:
    - Differentiable (soft assignment)
    - Reduces impact of outliers
    - Can capture piecewise patterns
    - Number of bins is controllable

    Limitations:
    - Loses fine-grained information
    - Choice of bin count affects results
    - May not preserve monotonic relationships
    - Not always beneficial for continuous data
    """

    def __init__(self, num_bins: int = 10, temperature: float = 1.0):
        """
        Args:
            num_bins: Number of discrete bins to create.
            temperature: Softmax temperature for bin assignment.
                        Lower values create harder, more discrete assignments.
        """
        super().__init__(name='discretize', op_type='transformer')
        self.num_bins = num_bins
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute bin edges
        min_val = x.min(dim=0, keepdim=True).values
        max_val = x.max(dim=0, keepdim=True).values

        # Create bin centers
        bin_centers = torch.linspace(0, 1, self.num_bins, device=x.device)

        # Normalize x to [0, 1]
        x_norm = (x - min_val) / (max_val - min_val + 1e-8)

        # Soft assignment to bins
        distances = torch.abs(x_norm.unsqueeze(-1) - bin_centers)
        weights = F.softmax(-distances / self.temperature, dim=-1)

        # Return weighted bin indices
        bin_indices = torch.arange(self.num_bins, device=x.device, dtype=x.dtype)
        soft_bins = (weights * bin_indices).sum(dim=-1)

        return soft_bins


# =============================================================================
# SOFT OPERATION MIXING
# =============================================================================

class SoftOperationMixer(nn.Module):
    """
    Soft Operation Mixer: Learn to combine multiple preprocessing operations.

    This module learns to mix multiple preprocessing operations using learnable
    weights. It enables differentiable operation selection, allowing the model
    to automatically learn which combination of preprocessing operations works best.

    The mixing is done via softmax-weighted combination, making the entire
    process differentiable and trainable via gradient descent.

    Use cases:
    - Want to learn optimal preprocessing pipeline
    - Multiple operations might be beneficial
    - Don't know which single operation is best
    - Ensemble-like preprocessing

    Advantages:
    - Differentiable and trainable
    - Can combine strengths of multiple operations
    - Automatic operation selection
    - Flexible temperature parameter controls hardness

    Limitations:
    - Adds extra trainable parameters
    - May be more computationally expensive
    - Requires careful tuning of temperature
    - Less interpretable than single operation
    """

    def __init__(self, operations: List[DifferentiableOperation]):
        """
        Args:
            operations: List of preprocessing operations to mix.
        """
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.num_ops = len(operations)

        # Learnable mixing weights
        self.weight_logits = nn.Parameter(torch.zeros(self.num_ops))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Apply soft mixture of operations.

        Args:
            x: Input tensor
            mask: Missing value mask
            temperature: Softmax temperature (lower = harder selection)

        Returns:
            Weighted combination of operation outputs
        """
        weights = F.softmax(self.weight_logits / temperature, dim=0)

        output = torch.zeros_like(x)
        for i, op in enumerate(self.operations):
            op_output = op(x, mask)
            output = output + weights[i] * op_output

        return output

    def get_selected_operation(self) -> Tuple[int, str]:
        """Return the operation with highest weight."""
        idx = self.weight_logits.argmax().item()
        return idx, self.operations[idx].name


# =============================================================================
# OPERATION LIBRARY
# =============================================================================

class OperationLibrary:
    """
    Library of all available preprocessing operations.

    Provides convenient access to all preprocessing operations organized by type:
    - Imputers: Handle missing values
    - Scalers: Normalize feature scales
    - Encoders: Convert categorical to numerical
    - Transformers: Create non-linear feature transformations

    Usage:
        imputers = OperationLibrary.get_imputers(num_features=10)
        scalers = OperationLibrary.get_scalers()
        all_ops = OperationLibrary.get_all_operations(num_features=10)
    """

    @staticmethod
    def get_imputers(num_features: int = None) -> Dict[str, DifferentiableOperation]:
        """
        Get all imputation operations.

        Args:
            num_features: If provided, includes LearnedImputer.

        Returns:
            Dictionary mapping names to imputation operations.
        """
        imputers = {
            'mean': MeanImputer(),
            'median': MedianImputer(),
            'knn': KNNImputer(k=5),
            'zero': ZeroImputer(),
        }
        if num_features is not None:
            imputers['learned'] = LearnedImputer(num_features)
        return imputers

    @staticmethod
    def get_scalers() -> Dict[str, DifferentiableOperation]:
        """
        Get all scaling operations.

        Returns:
            Dictionary mapping names to scaling operations.
        """
        return {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'quantile': QuantileScaler(),
            'log': LogScaler(),
            'identity': IdentityScaler(),
        }

    @staticmethod
    def get_encoders() -> Dict[str, DifferentiableOperation]:
        """
        Get all encoding operations.

        Returns:
            Dictionary mapping names to encoding operations.
        """
        return {
            'target': TargetEncoder(),
            'frequency': FrequencyEncoder(),
        }

    @staticmethod
    def get_transformers() -> Dict[str, DifferentiableOperation]:
        """
        Get all transformation operations.

        Returns:
            Dictionary mapping names to transformation operations.
        """
        return {
            'polynomial': PolynomialTransform(degree=2),
            'discretize': DiscretizeTransform(num_bins=10),
            'identity': IdentityScaler(),  # Reuse identity
        }

    @staticmethod
    def get_all_operations(num_features: int = None) -> List[DifferentiableOperation]:
        """
        Get all operations as a flat list.

        Args:
            num_features: If provided, includes LearnedImputer.

        Returns:
            List of all available preprocessing operations.
        """
        ops = []
        ops.extend(OperationLibrary.get_imputers(num_features).values())
        ops.extend(OperationLibrary.get_scalers().values())
        # Note: Encoders often need special handling, add selectively
        ops.extend(OperationLibrary.get_transformers().values())
        return ops
    
    @staticmethod
    def get_imputers(num_features: int = None) -> Dict[str, DifferentiableOperation]:
        imputers = {
            'mean': MeanImputer(),
            'median': MedianImputer(),
            'knn': KNNImputer(k=5),
            'zero': ZeroImputer(),
        }
        if num_features is not None:
            imputers['learned'] = LearnedImputer(num_features)
        return imputers
    
    @staticmethod
    def get_scalers() -> Dict[str, DifferentiableOperation]:
        return {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'quantile': QuantileScaler(),
            'log': LogScaler(),
            'identity': IdentityScaler(),
        }
    
    @staticmethod
    def get_encoders() -> Dict[str, DifferentiableOperation]:
        return {
            'target': TargetEncoder(),
            'frequency': FrequencyEncoder(),
        }
    
    @staticmethod
    def get_transformers() -> Dict[str, DifferentiableOperation]:
        return {
            'polynomial': PolynomialTransform(degree=2),
            'discretize': DiscretizeTransform(num_bins=10),
            'identity': IdentityScaler(),  # Reuse identity
        }
    
    @staticmethod
    def get_all_operations(num_features: int = None) -> List[DifferentiableOperation]:
        """Get all operations as a flat list."""
        ops = []
        ops.extend(OperationLibrary.get_imputers(num_features).values())
        ops.extend(OperationLibrary.get_scalers().values())
        # Note: Encoders often need special handling, add selectively
        ops.extend(OperationLibrary.get_transformers().values())
        return ops
