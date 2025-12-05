#!/usr/bin/env python3
"""
고급 Representation Similarity 메트릭 모듈
==========================================
논문에서 검증된 신경망 representation 비교 메트릭들

참조 논문:
1. CKA: Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)
2. SVCCA: Raghu et al., "SVCCA: Singular Vector Canonical Correlation Analysis" (NeurIPS 2017)
3. Procrustes: Ding et al., "Grounding Representation Similarity" (NeurIPS 2021)
4. RSA: Kriegeskorte et al., "Representational similarity analysis" (2008)
"""

import numpy as np
import torch
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


def centered_kernel_alignment(X, Y, kernel='linear'):
    """
    Centered Kernel Alignment (CKA)
    
    논문: Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)
    
    장점:
    - Orthogonal transformation에 invariant
    - Isotropic scaling에 invariant
    - 다른 initialization에도 robust
    
    Args:
        X: (n_samples, n_features_x)
        Y: (n_samples, n_features_y)
        kernel: 'linear' or 'rbf'
    
    Returns:
        float: CKA score (0~1, 1이 perfect match)
    """
    if kernel == 'linear':
        K = X @ X.T
        L = Y @ Y.T
    elif kernel == 'rbf':
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Center kernels
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Compute CKA
    hsic_xy = np.trace(K_centered @ L_centered)
    hsic_xx = np.trace(K_centered @ K_centered)
    hsic_yy = np.trace(L_centered @ L_centered)
    
    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    
    return float(cka)


def svcca_distance(X, Y, n_components=None):
    """
    Singular Vector Canonical Correlation Analysis (SVCCA)
    
    논문: Raghu et al., "SVCCA: Singular Vector Canonical Correlation Analysis" (NeurIPS 2017)
    
    장점:
    - Affine transformation에 invariant
    - Neuron permutation에 robust
    - 중요한 subspace 발견
    
    Args:
        X: (n_samples, n_features_x)
        Y: (n_samples, n_features_y)
        n_components: SVD components (default: min(n_samples, n_features) // 2)
    
    Returns:
        float: SVCCA distance (0~1, 0이 perfect match)
    """
    # 1. SVD for dimensionality reduction
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1], Y.shape[1]) // 2
    
    # Center data
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # SVD
    U_x, S_x, Vt_x = np.linalg.svd(X_centered, full_matrices=False)
    U_y, S_y, Vt_y = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Keep top components
    X_reduced = U_x[:, :n_components] * S_x[:n_components]
    Y_reduced = U_y[:, :n_components] * S_y[:n_components]
    
    # 2. CCA on reduced representations
    n_cca = min(n_components, X_reduced.shape[0] - 1)
    cca = CCA(n_components=n_cca)
    
    try:
        cca.fit(X_reduced, Y_reduced)
        X_c, Y_c = cca.transform(X_reduced, Y_reduced)
        
        # Compute canonical correlations
        correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_cca)]
        correlations = np.array(correlations)
        
        # Mean correlation (higher is more similar)
        mean_corr = np.mean(correlations)
        
        # Return distance (1 - correlation)
        return float(1.0 - mean_corr)
    except:
        # CCA failed, return max distance
        return 1.0


def procrustes_distance(X, Y):
    """
    Procrustes Distance
    
    논문: Ding et al., "Grounding Representation Similarity" (NeurIPS 2021)
    
    장점:
    - Geometrically intuitive
    - Orthogonal transformation (rotation, reflection)에 대해 optimal alignment
    - Consistent performance across benchmarks
    
    Args:
        X: (n_samples, n_features_x)
        Y: (n_samples, n_features_y)
    
    Returns:
        float: Procrustes distance (normalized)
    """
    # Center data
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Normalize
    X_norm = X_centered / np.linalg.norm(X_centered, 'fro')
    Y_norm = Y_centered / np.linalg.norm(Y_centered, 'fro')
    
    # Handle dimension mismatch
    min_dim = min(X_norm.shape[1], Y_norm.shape[1])
    if X_norm.shape[1] > min_dim:
        X_norm = X_norm[:, :min_dim]
    if Y_norm.shape[1] > min_dim:
        Y_norm = Y_norm[:, :min_dim]
    
    # Optimal rotation (Orthogonal Procrustes)
    U, _, Vt = np.linalg.svd(Y_norm.T @ X_norm)
    R = U @ Vt
    
    # Aligned X
    X_aligned = X_norm @ R.T
    
    # Procrustes distance
    distance = np.linalg.norm(Y_norm - X_aligned, 'fro')
    
    return float(distance)


def rsa_correlation(X, Y, metric='correlation'):
    """
    Representational Similarity Analysis (RSA)
    
    논문: Kriegeskorte et al., "Representational similarity analysis" (2008)
    
    장점:
    - 신경과학에서 검증됨
    - Model-agnostic
    - Interpretable
    
    Args:
        X: (n_samples, n_features_x)
        Y: (n_samples, n_features_y)
        metric: 'correlation' or 'euclidean'
    
    Returns:
        float: RSA correlation (0~1, 1이 perfect match)
    """
    # Compute Representational Dissimilarity Matrices (RDMs)
    if metric == 'correlation':
        # Correlation distance
        RDM_X = 1 - np.corrcoef(X)
        RDM_Y = 1 - np.corrcoef(Y)
    else:  # euclidean
        RDM_X = squareform(pdist(X, metric='euclidean'))
        RDM_Y = squareform(pdist(Y, metric='euclidean'))
    
    # Upper triangle (no diagonal)
    triu_indices = np.triu_indices_from(RDM_X, k=1)
    rdm_x_vec = RDM_X[triu_indices]
    rdm_y_vec = RDM_Y[triu_indices]
    
    # Spearman correlation between RDMs
    rsa_corr, _ = spearmanr(rdm_x_vec, rdm_y_vec)
    
    return float(rsa_corr)


def mutual_nearest_neighbors(X, Y, k=5):
    """
    Mutual Nearest Neighbors (MNN) metric
    
    개념: 얼마나 많은 샘플들이 두 representation space에서 서로 가까운지
    
    Args:
        X: (n_samples, n_features_x)
        Y: (n_samples, n_features_y)
        k: number of neighbors
    
    Returns:
        float: MNN score (0~1, 1이 perfect match)
    """
    n_samples = X.shape[0]
    
    # Pairwise distances
    dist_X = squareform(pdist(X, metric='euclidean'))
    dist_Y = squareform(pdist(Y, metric='euclidean'))
    
    # k-nearest neighbors in each space
    nn_X = np.argsort(dist_X, axis=1)[:, 1:k+1]  # Exclude self
    nn_Y = np.argsort(dist_Y, axis=1)[:, 1:k+1]
    
    # Count mutual nearest neighbors
    mnn_count = 0
    for i in range(n_samples):
        neighbors_X = set(nn_X[i])
        neighbors_Y = set(nn_Y[i])
        mnn_count += len(neighbors_X & neighbors_Y)
    
    # Normalize
    mnn_score = mnn_count / (n_samples * k)
    
    return float(mnn_score)


def linear_regression_score(X, Y):
    """
    Linear Regression Score
    
    개념: Y를 X로 얼마나 잘 예측할 수 있는지
    
    Args:
        X: (n_samples, n_features_x)
        Y: (n_samples, n_features_y)
    
    Returns:
        float: R^2 score (0~1, 1이 perfect fit)
    """
    # Add bias term
    X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
    
    # Solve least squares
    W, residuals, rank, s = np.linalg.lstsq(X_with_bias, Y, rcond=None)
    
    # Predict
    Y_pred = X_with_bias @ W
    
    # R^2 score
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return float(r2)


def compute_all_metrics(context1, context2, name1="Model1", name2="Model2"):
    """
    모든 similarity 메트릭 계산
    
    Args:
        context1: torch.Tensor or np.ndarray
        context2: torch.Tensor or np.ndarray
    
    Returns:
        dict: All metric results
    """
    # Convert to numpy if needed
    if isinstance(context1, torch.Tensor):
        context1 = context1.numpy()
    if isinstance(context2, torch.Tensor):
        context2 = context2.numpy()
    
    # Flatten to 2D (samples, features)
    X = context1.reshape(context1.shape[0], -1)
    Y = context2.reshape(context2.shape[0], -1)
    
    print(f"\n{'='*70}")
    print(f"고급 Similarity 메트릭 계산: {name1} vs {name2}")
    print(f"{'='*70}")
    print(f"Shape: X={X.shape}, Y={Y.shape}")
    
    metrics = {}
    
    # 1. CKA (Linear)
    print("\n  [1/8] Computing CKA (Linear)...")
    metrics['cka_linear'] = centered_kernel_alignment(X, Y, kernel='linear')
    
    # 2. CKA (RBF)
    print("  [2/8] Computing CKA (RBF)...")
    metrics['cka_rbf'] = centered_kernel_alignment(X, Y, kernel='rbf')
    
    # 3. SVCCA
    print("  [3/8] Computing SVCCA...")
    metrics['svcca_distance'] = svcca_distance(X, Y)
    metrics['svcca_similarity'] = 1.0 - metrics['svcca_distance']
    
    # 4. Procrustes Distance
    print("  [4/8] Computing Procrustes Distance...")
    metrics['procrustes_distance'] = procrustes_distance(X, Y)
    metrics['procrustes_similarity'] = 1.0 - metrics['procrustes_distance']
    
    # 5. RSA (Correlation)
    print("  [5/8] Computing RSA (Correlation)...")
    metrics['rsa_correlation'] = rsa_correlation(X, Y, metric='correlation')
    
    # 6. RSA (Euclidean)
    print("  [6/8] Computing RSA (Euclidean)...")
    metrics['rsa_euclidean'] = rsa_correlation(X, Y, metric='euclidean')
    
    # 7. Mutual Nearest Neighbors
    print("  [7/8] Computing MNN...")
    metrics['mnn_score'] = mutual_nearest_neighbors(X, Y, k=5)
    
    # 8. Linear Regression Score
    print("  [8/8] Computing Linear Regression Score...")
    metrics['linear_reg_r2'] = linear_regression_score(X, Y)
    
    print(f"\n{'='*70}")
    print("✅ 모든 메트릭 계산 완료!")
    print(f"{'='*70}")
    
    # Print summary
    print(f"\n  📊 고급 메트릭 결과:")
    print(f"     CKA (Linear):         {metrics['cka_linear']:.6f}")
    print(f"     CKA (RBF):            {metrics['cka_rbf']:.6f}")
    print(f"     SVCCA Similarity:     {metrics['svcca_similarity']:.6f}")
    print(f"     Procrustes Similarity: {metrics['procrustes_similarity']:.6f}")
    print(f"     RSA (Correlation):    {metrics['rsa_correlation']:.6f}")
    print(f"     RSA (Euclidean):      {metrics['rsa_euclidean']:.6f}")
    print(f"     MNN Score (k=5):      {metrics['mnn_score']:.6f}")
    print(f"     Linear Reg R²:        {metrics['linear_reg_r2']:.6f}")
    
    return metrics


def interpret_metrics(metrics):
    """
    메트릭 해석 및 권장사항
    """
    interpretation = []
    
    # CKA
    if metrics['cka_linear'] > 0.8:
        interpretation.append("✅ CKA (Linear): 매우 유사 (>0.8) - 선형 관계 강함")
    elif metrics['cka_linear'] > 0.5:
        interpretation.append("⚠️  CKA (Linear): 중간 유사도 (0.5~0.8)")
    else:
        interpretation.append("❌ CKA (Linear): 낮은 유사도 (<0.5)")
    
    # SVCCA
    if metrics['svcca_similarity'] > 0.7:
        interpretation.append("✅ SVCCA: 높은 subspace 정렬 (>0.7)")
    elif metrics['svcca_similarity'] > 0.4:
        interpretation.append("⚠️  SVCCA: 중간 subspace 정렬 (0.4~0.7)")
    else:
        interpretation.append("❌ SVCCA: 낮은 subspace 정렬 (<0.4)")
    
    # Procrustes
    if metrics['procrustes_similarity'] > 0.8:
        interpretation.append("✅ Procrustes: 매우 유사한 shape (>0.8)")
    elif metrics['procrustes_similarity'] > 0.5:
        interpretation.append("⚠️  Procrustes: 중간 shape 유사도 (0.5~0.8)")
    else:
        interpretation.append("❌ Procrustes: 다른 shape (<0.5)")
    
    # RSA
    if metrics['rsa_correlation'] > 0.7:
        interpretation.append("✅ RSA: 높은 구조적 유사도 (>0.7)")
    elif metrics['rsa_correlation'] > 0.4:
        interpretation.append("⚠️  RSA: 중간 구조적 유사도 (0.4~0.7)")
    else:
        interpretation.append("❌ RSA: 낮은 구조적 유사도 (<0.4)")
    
    return "\n".join(interpretation)


if __name__ == "__main__":
    # Test with random data
    print("Testing similarity metrics...")
    
    np.random.seed(42)
    X = np.random.randn(100, 512)
    
    # Similar representation
    Y_similar = X + np.random.randn(100, 512) * 0.1
    
    # Different representation
    Y_different = np.random.randn(100, 512)
    
    print("\n" + "="*70)
    print("Test 1: Similar Representations")
    print("="*70)
    metrics_sim = compute_all_metrics(X, Y_similar, "X", "Y_similar")
    print("\n해석:")
    print(interpret_metrics(metrics_sim))
    
    print("\n" + "="*70)
    print("Test 2: Different Representations")
    print("="*70)
    metrics_diff = compute_all_metrics(X, Y_different, "X", "Y_different")
    print("\n해석:")
    print(interpret_metrics(metrics_diff))
