import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ray import tune


# Define a function to train and evaluate the model
def train_model(config, checkpoint_dir=None):
    # Unpack the hyperparameters
    max_depth = config["max_depth"]
    n_estimators = config["n_estimators"]

    # Split the data
    # data = np.load(checkpoint_dir + "/data.npy")
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)

    # Report the accuracy metric
    tune.report(accuracy=accuracy)


# Define the search space for hyperparameters
search_space = {
    "max_depth": tune.randint(2, 10),
    "n_estimators": tune.choice([50, 100, 200])
}

# Define the data
data = {
    "X": np.random.rand(100, 10),
    "y": np.random.randint(0, 2, 100)
}

# Save data to a checkpoint directory
# np.save("./ray_results/hyperparameter_tuning/data.npy", data)

# Perform hyperparameter tuning using Ray Tune
analysis = tune.run(
    train_model,
    config=search_space,
    resources_per_trial={"cpu": 1},
    num_samples=10,
    verbose=1,
    local_dir="./ray_results",
    name="hyperparameter_tuning",
    stop={"training_iteration": 5},  # Stop after 5 iterations
    # checkpoint_freq=1,  # Save checkpoints after each iteration
    # checkpoint_at_end=True,  # Save checkpoint at the end
    resume=False  # Whether to resume from previous results
)

# Get the best hyperparameters
best_hyperparams = analysis.get_best_config(metric="accuracy", mode="max")
print("Best hyperparameters:", best_hyperparams)
