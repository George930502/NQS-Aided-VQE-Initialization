import torch
import torch.nn as nn

class FFNN(nn.Module):
    """
    A Feed-Forward Neural Network to model both amplitude and phase of a quantum state.
    The network outputs two values for each spin configuration:
    1. The logarithm of the wave function's amplitude (log|ψ|).
    2. The phase of the wave function (φ).
    log(ψ) = log|ψ| + i * φ
    """
    def __init__(self, n_visible, n_hidden, n_layers=2, device='cpu'):
        super(FFNN, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(n_visible, n_hidden))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
            
        # Output layer: 2 outputs for log-amplitude and phase
        layers.append(nn.Linear(n_hidden, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Set dtype for the network and move to device
        self.to(device=device, dtype=torch.float32)
        # Manually set parameter dtype to float32, as Sequential doesn't pass it
        for param in self.parameters():
            param.data = param.data.to(torch.float32)

    def log_prob(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates the complex logarithm of the wave function amplitude (log_psi).
        
        Args:
            sigma: A tensor of shape [n_batch, n_visible] with {-1, 1} spins.
        
        Returns:
            A complex tensor of shape [n_batch] where each element is log(ψ).
        """
        # Ensure input is float32 for the network
        sigma_float = sigma.to(dtype=torch.float32)
        
        # network_output shape: [n_batch, 2]
        network_output = self.network(sigma_float)
        
        # Unpack the two outputs
        log_amplitude = network_output[:, 0]
        phase = network_output[:, 1]
        
        # Combine into a complex number: log(ψ) = log|ψ| + i*φ
        return log_amplitude + 1j * phase

class RBM(torch.nn.Module):
    """
    Restricted Boltzmann Machine (RBM) for modeling the quantum wave function.
    """
    def __init__(self, n_visible, n_hidden, device='cpu'):
        super(RBM, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(n_visible, dtype=torch.cfloat, device=device) * 0.05)
        self.b = torch.nn.Parameter(torch.randn(n_hidden, dtype=torch.cfloat, device=device) * 0.05)
        self.W = torch.nn.Parameter(torch.randn(n_visible, n_hidden, dtype=torch.cfloat, device=device) * 0.05)
        self.to(device)

    def log_prob(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logarithm of the unnormalized probability (log-amplitude) 
        for a batch of spin configurations.
        
        Args:
            sigma: A tensor of shape [n_batch, n_visible]
            
        Returns:
            A tensor of shape [n_batch] containing the log-amplitude for each configuration.
        """
        sigma_complex = sigma.to(dtype=torch.cfloat)
        
        # visible_term shape: [n_batch]
        visible_term = torch.einsum('bi,i->b', sigma_complex, self.a)
        
        # theta shape: [n_batch, n_hidden]
        theta = torch.einsum('bi,ij->bj', sigma_complex, self.W) + self.b
        
        # hidden_term shape: [n_batch]
        hidden_term = torch.sum(torch.log(2 * torch.cosh(theta)), dim=1)
        
        return visible_term + hidden_term