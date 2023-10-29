import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 9.8
v0 = 1000
k = 0.003
dt = 0.01

def compute_trajectory(theta_deg):
    theta = np.radians(theta_deg)
    x, y = [0], [0]
    vx, vy = v0 * np.cos(theta), v0 * np.sin(theta)

    while y[-1] >= 0:
        ax = -k * vx * np.sqrt(vx**2 + vy**2)
        ay = -g - k * vy * np.sqrt(vx**2 + vy**2)

        vx += ax * dt
        vy += ay * dt

        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)

    return x, y

def animate(i):
    ax.clear()
    ax.plot(x[:i], y[:i], color='b')
    ax.scatter(x[i], y[i], color='r', s=100)
    ax.text(0.1, 0.9, f"height: {y[i]:.2f}m\n distance: {x[i]:.2f}m", transform=ax.transAxes)
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("height (m)")
    ax.grid(True)


angles_search = np.linspace(0, 90, 500)
max_distance = 0
best_angle = 0

for angle in angles_search:
    x_temp, y_temp = compute_trajectory(angle)
    if x_temp[-1] > max_distance:
        max_distance = x_temp[-1]
        best_angle = angle

print(f"max length：{max_distance:.2f}m，max length angle：{best_angle:.2f}°")
x, y = compute_trajectory(best_angle)

fig, ax = plt.subplots(figsize=(10, 5))
ani = animation.FuncAnimation(fig, animate, frames=len(x), repeat=False, interval=100)
plt.show()
