import jax
import jax.numpy as jnp
from evosax import CMA_ES


# Define a simple function to optimize
def sphere(x):
    return jnp.sum(x ** 2)


# Set up the problem
num_dims = 5
popsize = 50
strategy = CMA_ES(popsize=popsize, num_dims=num_dims)
es_params = strategy.default_params

# Initialize the strategy
rng = jax.random.PRNGKey(0)
state = strategy.initialize(rng)

# Run the optimization for 100 generations
for gen in range(100):
    rng, rng_ask, rng_eval = jax.random.split(rng, 3)

    # Ask for new solutions
    x, state = strategy.ask(rng_ask, state, es_params)

    # Evaluate the solutions
    fitness = jax.vmap(sphere)(x)

    # Tell the results back to the strategy
    state = strategy.tell(x, fitness, state, es_params)

    # Print progress
    if gen % 10 == 0:
        best_fitness = jnp.min(fitness)
        print(f"Generation {gen}: Best fitness = {best_fitness}")

# Get the best solution
best_solution = state.mean  # The mean of the distribution is typically the best estimate
best_fitness = sphere(best_solution)

print(f"\nBest solution: {best_solution}")
print(f"Best fitness: {best_fitness}")