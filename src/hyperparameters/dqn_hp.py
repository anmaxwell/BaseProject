batch_size = 32

gamma = 0.99

eps_start=1.0
eps_decay = 0.995
eps_min = 0.1      # Minimal exploration rate (epsilon-greedy)

num_rounds = 150
learning_limit = 100
replay_limit = 1000  # Number of steps until starting replay