import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image
import heapq
import math

# -------------------- A* PATHFINDING CORE --------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(array, start, goal):
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]

        close_set.add(current)

        for dx, dy in neighbors:
            neighbor = current[0] + dx, current[1] + dy

            if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
                if array[neighbor[0]][neighbor[1]] > 8:
                    continue
            else:
                continue

            tentative_g_score = gscore[current] + (math.sqrt(2) if dx != 0 and dy != 0 else 1)

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] > 8:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

# -------------------- RADIATION SIMULATION --------------------
def generate_radiation_grid(grid_size, center, max_radiation=10):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    cx, cy = center
    cx /= grid_size
    cy /= grid_size

    sigma = 0.28
    gauss = max_radiation * np.exp(-((X - cy) ** 2 + (Y - cx) ** 2) / (2 * sigma ** 2))
    noise = np.random.normal(0, 0.3, (grid_size, grid_size))
    radiation = np.clip(gauss + noise, 0, None)

    return radiation

# -------------------- MAIN APP GUI --------------------
class RadiationApp:
    def __init__(self, master):
        self.master = master

        master.title("UAV Radiation Pathfinding Simulation")

        self.grid_size = 200
        self.image_width = 1685
        self.image_height = 955
        self.radiation_center = None
        self.start = None
        self.goal = None
        self.background_image = None
        self.img_extent = (0, self.image_width, 0, self.image_height)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack()

        self.btn_bg = tk.Button(master, text="Load Background Image", command=self.load_background)
        self.btn_bg.pack()

        self.btn_clear = tk.Button(master, text="Reset Simulation", command=self.reset_simulation)
        self.btn_clear.pack()

        self.btn_save = tk.Button(master, text="Save Final Image", command=self.save_figure)
        self.btn_save.pack()

        self.canvas.mpl_connect("button_press_event", self.onclick)

        self.status_label = tk.Label(master, text="Click to mark Radiation Source (Upper Corner)")
        self.status_label.pack()

    def load_background(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path).resize((self.image_width, self.image_height))
            self.background_image = np.array(img)
            self.ax.clear()
            self.ax.imshow(self.background_image, extent=self.img_extent, origin='lower')
            self.status_label.config(text="Background loaded. Click to mark Radiation Source")
            self.canvas.draw()

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        gx = int((x / self.image_width) * self.grid_size)
        gy = int((y / self.image_height) * self.grid_size)

        if not self.radiation_center:
            self.radiation_center = (gy, gx)
            self.status_label.config(text="Marked Radiation Source. Now click to mark Drone Start Point")
            self.ax.plot(x, y, marker='X', color='red', markersize=12, label='Radiation Source')

        elif not self.start:
            self.start = (gy, gx)
            self.status_label.config(text="Marked Drone Start. Now click to mark User Destination")
            self.ax.plot(x, y, marker='^', color='blue', markersize=12, label='Drone Start')

        elif not self.goal:
            self.goal = (gy, gx)
            self.status_label.config(text="Marked User. Calculating path...")
            self.ax.plot(x, y, marker='o', color='green', markersize=10, label='User')
            self.run_simulation()

        self.canvas.draw()

    def run_simulation(self):
        radiation_grid = generate_radiation_grid(self.grid_size, center=self.radiation_center)

        extent = self.img_extent
        contour = self.ax.contourf(np.linspace(extent[0], extent[1], self.grid_size),
                                   np.linspace(extent[2], extent[3], self.grid_size),
                                   radiation_grid,
                                   levels=30, cmap='YlOrRd', alpha=0.6)

        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception as e:
                print(f"Warning while removing colorbar: {e}")
            self.cbar = None

        self.cbar = self.fig.colorbar(contour, ax=self.ax)
        self.cbar.set_label("Radiation Level (μSv/h)")

        path = astar(radiation_grid, self.start, self.goal)
        if path:
            self.status_label.config(text="analysing path...")

            px, py = zip(*path)
            px = [int((y / self.grid_size) * self.image_height) for y in px]
            py = [int((x / self.grid_size) * self.image_width) for x in py]

            def draw_step(i=0):
                if i < len(px) - 1:
                    self.ax.plot(py[i:i+2], px[i:i+2], color='blue', linewidth=2)
                    self.canvas.draw()
                    self.master.after(10, draw_step, i + 1)
                else:
                    self.ax.plot(py, px, color='blue', linewidth=2, label='Safe Path')
                    self.status_label.config(text="Path calculated!")
                    self.ax.legend()
                    self.canvas.draw()

            draw_step()
        else:
            self.status_label.config(text="No path found! Danger too high!")

    def reset_simulation(self):
        self.radiation_center = None
        self.start = None
        self.goal = None

        self.ax.clear()

        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception as e:
                print(f"Warning while removing colorbar: {e}")
            self.cbar = None

        if self.background_image is not None:
            self.ax.imshow(self.background_image, extent=self.img_extent, origin='lower')

        if self.ax.get_legend():
            self.ax.legend_.remove()

        self.status_label.config(text="Simulation reset. Click to mark Radiation Source")
        self.canvas.draw()

    def save_figure(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            self.fig.savefig(file_path, dpi=300)
            self.status_label.config(text=f"Image saved to {file_path}")

# -------------------- LAUNCH APP --------------------
root = tk.Tk()
app = RadiationApp(root)
root.mainloop()
