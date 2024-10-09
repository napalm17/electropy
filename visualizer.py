from matplotlib import cm

from system import System
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from charges import Charge
from scipy.constants import c


class Visualizer:

    def __init__(self, system: System):
        self.system = system

    @property
    def time(self) -> float:
        return self.system.time

    @property
    def field_dict(self) -> dict:
        return self.system.field_dict

    @property
    def charges(self) -> list[Charge]:
        return self.system.charges  # Returns the charges in the system

    def evolve_system(self, t: float, dt=0.01) -> None:
        self.system.evolve_by(t, dt)

    def plot_field(self, field_type: str, pos: np.ndarray, time: float) -> None:
        """Plot the specified field at given positions and time."""
        # Get the field values at the specified positions and time
        field_values = self.field_dict[field_type](pos, time)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                  field_values[:, 0], field_values[:, 1], field_values[:, 2],
                  length=0.6, linewidth=1, normalize=True, color='red', alpha=0.6)

        charge_positions = [charge.position(time) for charge in self.charges]
        ax.scatter(*zip(*charge_positions), color='blue', s=100)  # Unzip coordinates
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'{field_type.capitalize()} Field at Time {time}')
        plt.show()

    def plotly_field(self, field_type: str, time: float, pos: np.ndarray) -> None:
        """Plot the specified field at given positions and time using Plotly."""
        field_values = self.field_dict[field_type](pos, time)
        field_values = np.arctan(5*field_values)
        fig = go.Figure(data=go.Cone(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            u=field_values[:, 0], v=field_values[:, 1], w=field_values[:, 2],
            sizemode="scaled", sizeref=0.75, anchor="tail", colorscale=[[0, 'black'], [1, 'red']], showscale=True))

        charge_positions = [charge.position(time) for charge in self.charges]
        fig.add_trace(go.Scatter3d(
            x=charge_positions[:, 0],
            y=charge_positions[:, 1],
            z=charge_positions[:, 2],
            mode='markers',
            marker=dict(size=20, color='blue')
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                aspectmode='data'
            ),
            title=f'{field_type.capitalize()} Field at Time {time}',
            width=800,
            height=800
        )
        # Show the figure
        fig.show()

    def animate(self, field_type: str, time_interval: tuple, pos: np.ndarray, dt=1e-3) -> None:
        """Animate the specified field and charge positions over a time interval."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        initial_field_values = self.field_dict[field_type](pos, time_interval[0])
        quiver = ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                           initial_field_values[:, 0], initial_field_values[:, 1], initial_field_values[:, 2],
                           length=0.6, color='red', alpha=0.6)
        charge_positions = np.array([charge.position(time_interval[0]) for charge in self.charges])
        colors =['yellow' if charge.magnitude < 0 else 'red' for charge in self.charges]
        scatter = ax.scatter(charge_positions[:, 0], charge_positions[:, 1], charge_positions[:, 2], c=colors, s=100)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'{field_type.capitalize()} Field Animation')

        def update(frame):
            nonlocal quiver
            time_step = time_interval[0] + frame * dt  # Update time based on frame
            # Get new field values
            field_values = self.field_dict[field_type](pos, time_step)
            field_values = np.arctan(0.5*field_values)
            field_values /= np.max(field_values)
            magnitudes = np.linalg.norm(field_values, axis=-1)
            colormap = cm.get_cmap('plasma_r')  # You can choose any colormap
            colors = colormap(magnitudes)

            quiver.remove()  # Remove previous quiver
            quiver = ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                               field_values[:, 0], field_values[:, 1], field_values[:, 2],
                               length=0.4, color=colors, alpha=0.5)
            charge_positions = np.array([charge.position(time_step) for charge in self.charges])
            scatter._offsets3d = (charge_positions[:, 0], charge_positions[:, 1], charge_positions[:, 2])
            ax.set_title(f'{field_type.capitalize()} Field Animation, t={time_step}')
            return quiver, scatter

        num_frames = int((time_interval[1] - time_interval[0]) / dt)
        ani = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=1)
        plt.show()




