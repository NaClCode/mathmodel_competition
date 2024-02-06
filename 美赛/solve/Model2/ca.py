import numpy as np

alpha = 0.2
c = 100
beta = 0.01
sigma = 0.01
grid_size = (200, 200)
initial_population = 50
theta = 0.01

grid = np.ones(grid_size)

population = np.zeros(grid_size)
start_location = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
population[start_location] = initial_population

iterations = 100

for iteration in range(iterations):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if population[i, j] > 0:
                Z_m = alpha * grid[i, j] + 0.5
                B = Z_m * (1 - Z_m) * c
                population[i, j] = population[i, j] + B - theta 
                if population[i, j] < 0:
                    population[i, j] = 0
                grid[i, j] = grid[i, j] - population[i, j] * beta + sigma
                if grid[i, j] < 0.02 * population[i, j]:  
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_x, new_y = i + dx, j + dy
                        if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1] and grid[new_x, new_y] > 0:
                            population[new_x, new_y] += 0.1

    print(sum(grid.sum(axis=0)))
    print(sum(population.sum(axis=1)))

# Print or visualize the final state of the grid and population array if needed
print("Final population:")
print(population)
print("Final resource grid:")
print(grid)
