import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.2
c = 2
beta = 0.01
sigma = 0.1
grid_size = (20, 20)
initial_population = 1
# 动画显示
fig, ax = plt.subplots()
population = np.zeros(grid_size)
start_location = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
population[start_location] = initial_population
grid = np.ones(grid_size) 
iterations = 100

def update(frame):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if population[i, j] > 0 and grid[i, j] > 0:
                Z_m = alpha * grid[i, j] + 0.5
                B = Z_m * (1 - Z_m) * c
                grid[i, j] = grid[i, j] - population[i, j] * beta + 0.1
                if grid[i, j] < 0:
                    grid[i, j] = 0
                    population[i, j] = 0
                if population[i, j] > 0:
                    population[i, j] = population[i, j] + B + grid[i, j] - 1
                    print(grid[i, j])
                if grid[i, j] < 0.8:  
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_x, new_y = i + dx, j + dy
                        if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1] and grid[new_x, new_y] > 0:
                            population[new_x, new_y] += 1
            elif grid[i, j] < 0:
                grid[i, j] = 0
    ax.clear()

    #print(grid)
    ax.imshow(population, cmap='winter')
    ax.axis('off')
    #ax.grid(True)
    ax.set_title(f'Population Step {frame}')

# 模拟演化，并绘制演化过程
ani = animation.FuncAnimation(fig, update, frames=100, interval=200)
plt.show()