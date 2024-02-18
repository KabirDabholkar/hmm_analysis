import torch
import torch.nn as nn
import torch.optim as optim


class HMM(nn.Module):
    def __init__(self, num_hidden_states, num_observation_states):
        super(HMM, self).__init__()
        self.num_hidden_states = num_hidden_states
        self.num_observation_states = num_observation_states

        # Transition matrix A
        self.A = nn.Parameter(torch.rand(num_hidden_states, num_hidden_states))
        self.A.data = torch.log(self.A.data / self.A.sum(dim=1, keepdim=True))

        # Emission matrix B
        self.B = nn.Parameter(torch.rand(num_hidden_states, num_observation_states))
        self.B.data = torch.log(self.B.data / self.B.sum(dim=1, keepdim=True))

        # Initial state probabilities pi
        self.pi = nn.Parameter(torch.rand(num_hidden_states))
        self.pi.data = torch.log(self.pi.data / self.pi.sum())

    def forward(self, observations):
        T = observations.size(0)
        N = self.num_hidden_states
        alpha = torch.zeros(T, N)

        # Forward algorithm
        for t in range(T):
            if t == 0:
                alpha[t] = self.pi + self.B[:, observations[t]]
            else:
                alpha[t] = torch.logsumexp(alpha[t - 1].unsqueeze(1) + self.A, dim=0) + self.B[:, observations[t]]

        return torch.logsumexp(alpha[-1], dim=0)


# Example usage:

# Define the model
num_hidden_states = 2
num_observation_states = 3
model = HMM(num_hidden_states, num_observation_states)

# Generate some dummy observations
observations = torch.tensor([0, 1, 2, 1, 0, 2])

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    log_likelihood = -model(observations)
    log_likelihood.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Log Likelihood: {-log_likelihood.item()}')

# After training, you can use the model for inference
