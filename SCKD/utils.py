# @Author:  ThanhNX
import torch
import numpy as np



def get_T_distribution(X : torch.Tensor, df : int = 3):
    """
    X : torch.Tensor
        The input data tensor , shape of (n_samples, n_features)
    df : int (default = 3)
        The degrees of freedom parameter for the t-distribution
    Returns
    -------
    t_distribution : torch.distributions.StudentT
        The multivariate t-distribution object
    """
    assert len(X.shape) == 2, "The input tensor should be 2D"
    X_np = X.cpu().numpy()
    # Calculate mean and covariance matrix
    mean = torch.mean(X, dim=0)
    covariance = torch.diag(torch.var(X, dim=0))
    # Degrees of freedom parameter for t-distribution
    df = 2  # Adjust as needed based on your data and assumptions
    # Create a multivariate t-distribution
    t_distribution = torch.distributions.StudentT(df = df, loc=mean, scale = torch.diagonal(covariance).sqrt())
    return t_distribution


def get_normal_distribution(X : torch.Tensor):
    """
    X : torch.Tensor
        The input data tensor , shape of (n_samples, n_features)
    Returns
    -------
    normal_distribution : torch.distributions.MultivariateNormal
        The multivariate normal distribution object
    """
    assert len(X.shape) == 2, "The input tensor should be 2D"
    X_np = X.numpy()
    # Calculate mean and covariance matrix
    mean = torch.mean(X, dim=0)
    covariance = torch.diag(torch.var(X, dim=0))
    # Create a multivariate normal distribution
    normal_distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance)
    return normal_distribution
