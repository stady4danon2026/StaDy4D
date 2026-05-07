"""
Visualize the different camera trajectory types
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import trajectory functions
import sys
sys.path.insert(0, '.')
from data_generate_carla import (
    generate_orbit_trajectory,
    generate_forward_trajectory,
    generate_drone_trajectory
)


def plot_trajectory_3d(ax, trajectory, title, color='blue'):
    """Plot a 3D trajectory"""
    # Extract positions
    positions = np.array([(pose[0], pose[1], pose[2]) for pose in trajectory])
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Plot trajectory line
    ax.plot(x, y, z, color=color, linewidth=2, alpha=0.7, label='Camera path')

    # Plot start and end points
    ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, marker='o', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=100, marker='X', label='End')

    # Plot direction indicators (every 30 frames)
    for i in range(0, len(trajectory), 30):
        pose = trajectory[i]
        if len(pose) == 6:
            px, py, pz, pitch, yaw, roll = pose
        else:
            px, py, pz, yaw = pose
            pitch = -20

        # Calculate look direction
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        dx = np.cos(pitch_rad) * np.cos(yaw_rad)
        dy = np.cos(pitch_rad) * np.sin(yaw_rad)
        dz = np.sin(pitch_rad)

        # Draw arrow
        ax.quiver(px, py, pz, dx, dy, dz,
                 length=3, color='orange', alpha=0.6, arrow_length_ratio=0.3)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, mid_z + max_range)

    ax.grid(True, alpha=0.3)


def plot_trajectory_topdown(ax, trajectory, title, color='blue'):
    """Plot a top-down (2D) view of trajectory"""
    positions = np.array([(pose[0], pose[1]) for pose in trajectory])
    x, y = positions[:, 0], positions[:, 1]

    # Plot trajectory
    ax.plot(x, y, color=color, linewidth=2, alpha=0.7)
    ax.scatter([x[0]], [y[0]], color='green', s=100, marker='o', label='Start')
    ax.scatter([x[-1]], [y[-1]], color='red', s=100, marker='X', label='End')

    # Plot direction arrows (every 30 frames)
    for i in range(0, len(trajectory), 30):
        pose = trajectory[i]
        px, py = pose[0], pose[1]

        if len(pose) == 6:
            yaw = pose[4]
        else:
            yaw = pose[3]

        yaw_rad = np.radians(yaw)
        dx = np.cos(yaw_rad)
        dy = np.sin(yaw_rad)

        ax.arrow(px, py, dx*2, dy*2, head_width=1, head_length=1,
                fc='orange', ec='orange', alpha=0.6)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')


def visualize_all_trajectories():
    """Create visualizations for all trajectory types"""
    frames = 150

    # Generate trajectories
    print("Generating trajectories...")
    orbit_traj = generate_orbit_trajectory(0, frames_per_video=frames)
    forward_traj = generate_forward_trajectory(0, frames_per_video=frames)
    drone_traj = generate_drone_trajectory(0, frames_per_video=frames)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # 3D plots (top row)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_trajectory_3d(ax1, orbit_traj, 'Orbit Trajectory (3D)', color='#2E86AB')

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_trajectory_3d(ax2, forward_traj, 'Forward Trajectory (3D)', color='#A23B72')

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_trajectory_3d(ax3, drone_traj, 'Drone Trajectory (3D)', color='#F18F01')

    # Top-down plots (bottom row)
    ax4 = fig.add_subplot(2, 3, 4)
    plot_trajectory_topdown(ax4, orbit_traj, 'Orbit Trajectory (Top View)', color='#2E86AB')

    ax5 = fig.add_subplot(2, 3, 5)
    plot_trajectory_topdown(ax5, forward_traj, 'Forward Trajectory (Top View)', color='#A23B72')

    ax6 = fig.add_subplot(2, 3, 6)
    plot_trajectory_topdown(ax6, drone_traj, 'Drone Trajectory (Top View)', color='#F18F01')

    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: trajectory_comparison.png")

    # Print statistics
    print("\n" + "="*60)
    print("Trajectory Statistics")
    print("="*60)

    for name, traj in [("Orbit", orbit_traj), ("Forward", forward_traj), ("Drone", drone_traj)]:
        positions = np.array([(p[0], p[1], p[2]) for p in traj])
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)

        print(f"\n{name} Trajectory:")
        print(f"  Total path length: {total_distance:.2f} meters")
        print(f"  Height range: {positions[:, 2].min():.2f}m - {positions[:, 2].max():.2f}m")
        print(f"  Avg speed: {total_distance / frames:.3f} m/frame")


def visualize_multiple_videos():
    """Show how different video indices create different paths"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    trajectory_types = [
        ('orbit', 'Orbit'),
        ('forward', 'Forward'),
        ('drone', 'Drone')
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    for idx, (traj_type, title) in enumerate(trajectory_types):
        ax = axes[idx]

        # Generate 5 videos worth of trajectories
        for video_idx in range(5):
            if traj_type == 'orbit':
                traj = generate_orbit_trajectory(video_idx, frames_per_video=150)
            elif traj_type == 'forward':
                traj = generate_forward_trajectory(video_idx, frames_per_video=150)
            else:  # drone
                traj = generate_drone_trajectory(video_idx, frames_per_video=150)

            positions = np.array([(p[0], p[1]) for p in traj])
            ax.plot(positions[:, 0], positions[:, 1],
                   color=colors[video_idx], linewidth=2, alpha=0.7,
                   label=f'Video {video_idx}')
            ax.scatter([positions[0, 0]], [positions[0, 1]],
                      color=colors[video_idx], s=50, marker='o')

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'{title}: All 5 Videos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()
    plt.savefig('trajectory_videos_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: trajectory_videos_comparison.png")


if __name__ == "__main__":
    print("="*60)
    print("Camera Trajectory Visualization")
    print("="*60)

    # Visualize all trajectory types
    visualize_all_trajectories()

    # Visualize multiple videos per type
    print("\n" + "="*60)
    print("Comparing All 5 Videos Per Trajectory Type")
    print("="*60)
    visualize_multiple_videos()

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
