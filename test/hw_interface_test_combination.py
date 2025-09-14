import rospy
from geometry_msgs.msg import Pose
import numpy as np
from hw_interface import HwInterface  # Assume the HwInterface class is in a file named hw_interface.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import alpha, beta, lognorm
import json
from envs.Locate import Locate
from envs.panda_transformer import PandaTransformer

# Initialize the hardware interface with panda, camera, and gazebo enabled
hw_interface = HwInterface(panda=True, cameras=["/image_raw"], gazebo=True)

# Test get_component_info method
components = hw_interface.get_component_info()
print("Components:", components)

# Test get_robot_position method
robot_position = hw_interface.get_robot_position()
print("Robot Position:", robot_position)

# Test get_gripper_position method
gripper_position = hw_interface.get_gripper_position()
print("Gripper Position:", gripper_position)

my_trans = PandaTransformer()
my_trans.init_pose()

locate = Locate()
base_coordinates, relative_position = locate.locate_colored_object(colormode='cube')
print("Object position:", base_coordinates)

# Function to sample new x, y coordinates based on a specified distribution
def sample_new_position(x, y, distribution="normal", num_samples=50, **kwargs):
    if distribution == "normal":
        std_dev = kwargs.get("std_dev", 0.01)
        sampled_x = np.random.normal(x, std_dev, num_samples)
        sampled_y = np.random.normal(y, std_dev, num_samples)
    elif distribution == "triangular":
        left = kwargs.get("left", x - 0.005)
        mode = kwargs.get("mode", x)
        right = kwargs.get("right", x + 0.005)
        sampled_x = np.random.triangular(left, mode, right, num_samples)
        left = kwargs.get("left", y - 0.005)
        mode = kwargs.get("mode", y)
        right = kwargs.get("right", y + 0.005)
        sampled_y = np.random.triangular(left, mode, right, num_samples)
    elif distribution == "alpha":
        a = kwargs.get("a", 2)
        sampled_x = alpha.rvs(a, loc=x, scale=0.004, size=num_samples)
        sampled_y = alpha.rvs(a, loc=y, scale=0.004, size=num_samples)
    elif distribution == "beta":
        a = kwargs.get("a", 2)
        b = kwargs.get("b", 5)
        sampled_x = beta.rvs(a, b, loc=x, scale=0.001, size=num_samples)
        sampled_y = beta.rvs(a, b, loc=y, scale=0.001, size=num_samples)
    elif distribution == "lognormal":
        mean = kwargs.get("mean", 0)
        sigma = kwargs.get("sigma", 0.25)
        sampled_x = lognorm.rvs(sigma, loc=x, scale=np.exp(mean), size=num_samples)
        sampled_y = lognorm.rvs(sigma, loc=y, scale=np.exp(mean), size=num_samples)
    elif distribution == "uniform":
        low = kwargs.get("low", x - 0.02)
        high = kwargs.get("high", x + 0.02)
        sampled_x = np.random.uniform(low, high, num_samples)
        low = kwargs.get("low", y - 0.02)
        high = kwargs.get("high", y + 0.02)
        sampled_y = np.random.uniform(low, high, num_samples)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return sampled_x, sampled_y

# Function to sample new gripper widths based on a specified distribution
def sample_gripper_widths(mean_width, distribution="uniform", num_samples=50, **kwargs):
    if distribution == "normal":
        std_dev = kwargs.get("std_dev", 0.01)
        sampled_widths = np.random.normal(mean_width, std_dev, num_samples)
    elif distribution == "triangular":
        left = kwargs.get("left", mean_width - 0.005)
        mode = kwargs.get("mode", mean_width)
        right = kwargs.get("right", mean_width + 0.005)
        sampled_widths = np.random.triangular(left, mode, right, num_samples)
    elif distribution == "alpha":
        a = kwargs.get("a", 2)
        sampled_widths = alpha.rvs(a, loc=mean_width, scale=0.001, size=num_samples)
    elif distribution == "beta":
        a = kwargs.get("a", 3)
        b = kwargs.get("b", 0.5)
        sampled_widths = beta.rvs(a, b, loc=mean_width, scale=0.0045, size=num_samples)
    elif distribution == "lognormal":
        mean = kwargs.get("mean", 0)
        sigma = kwargs.get("sigma", 0.0001)
        sampled_widths = lognorm.rvs(sigma, loc=mean_width, scale=np.exp(mean), size=num_samples)
    elif distribution == "uniform":
        sampled_widths = np.random.uniform(mean_width - 0.01, mean_width + 0.01, num_samples)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return sampled_widths

# Choose distribution and sample new positions for x and y
position_distribution_type = "normal"  # Change this to use a different distribution
sampled_x, sampled_y = sample_new_position(base_coordinates[0], base_coordinates[1], distribution=position_distribution_type)
print("Sampled position:", sampled_x, sampled_y)

# Set mean gripper width and sample new widths
mean_gripper_width = 0.03  # Assuming the nominal width for grasping is 0.03 meters
width_distribution_type = "uniform"  # Change this to use a different distribution
sampled_gripper_widths = sample_gripper_widths(mean_gripper_width, distribution=width_distribution_type)
print("Sampled gripper widths:", sampled_gripper_widths)

# Convert initial positions to lists if they are numpy arrays
initial_robot_position = robot_position.tolist() if isinstance(robot_position, np.ndarray) else robot_position
initial_gripper_position = gripper_position.tolist() if isinstance(gripper_position, np.ndarray) else gripper_position

# List to store data
data_log = {
    "initial_positions": {
        "robot_position": initial_robot_position,
        "gripper_position": initial_gripper_position,
    },
    "iterations": []
}

# Lists to store data for plotting
iteration_numbers = []
selected_positions = []
sampled_widths = []
grasp_results = []

# Select uncertainty type: position, width, or both
add_uncertainty_to = "both"  # Change this to "position", "width", or "both"

for i in range(50):
    selected_x, selected_y, sampled_gripper_width = base_coordinates[0], base_coordinates[1], mean_gripper_width

    if add_uncertainty_to in ["position", "both"]:
        selected_x = float(sampled_x[i])
        selected_y = float(sampled_y[i])
        print(f"Iteration: {i + 1}, Selected position for action: x={selected_x}, y={selected_y}")

    if add_uncertainty_to in ["width", "both"]:
        sampled_gripper_width = sampled_gripper_widths[i]
        print(f"Iteration: {i + 1}, Sampled gripper width for action: {sampled_gripper_width}")

    iteration_data = {
        "iteration": i + 1,
        "selected_position": {"x": selected_x, "y": selected_y},
        "sampled_gripper_width": sampled_gripper_width,
        "actions": []
    }

    # Open the gripper
    hw_interface.move_gripper()
    iteration_data["actions"].append({"action": "move_gripper", "status": "opened"})

    # Move the gripper to the selected position
    success = hw_interface.move_cartesian(x=selected_x, y=selected_y, z=0.2, o_x=1, o_y=0, o_z=0, o_w=0, vel=0.1, acc=0.1)
    print("Move Cartesian:", "Success" if success else "Failed")
    hw_interface.move_cartesian(z=0.02, vel=0.1, acc=0.1)

    # Grasp the object
    success = hw_interface.grasp_object(width=sampled_gripper_width)
    iteration_data["actions"].append({"action": "grasp_object", "status": "Success" if success else "Failed"})
    print("Grasping object:", "Success" if success else "Failed")

    # Move the gripper up after grasping
    hw_interface.move_cartesian(z=0.2, vel=0.1, acc=0.1)
    hw_interface.move_cartesian(x=0.6, y=0.05, vel=0.1, acc=0.1)
    hw_interface.move_cartesian(x=base_coordinates[0], y=base_coordinates[1], vel=0.1, acc=0.1)
    hw_interface.move_cartesian(z=0.02, vel=0.1, acc=0.1)
    hw_interface.move_gripper()
    hw_interface.move_cartesian(z=0.2, vel=0.1, acc=0.1)
    hw_interface.move_cartesian(x=0.2, y=0.1, vel=0.1, acc=0.1)
    iteration_numbers.append(i + 1)
    selected_positions.append((selected_x, selected_y))
    sampled_widths.append(sampled_gripper_width)
    grasp_results.append(success)
# Log iteration data
    data_log["iterations"].append(iteration_data)

# Calculate success and failure rates
success_rate = sum(grasp_results) / len(grasp_results)
failure_rate = 1 - success_rate
print(f"Success Rate: {success_rate * 100:.2f}%")
print(f"Failure Rate: {failure_rate * 100:.2f}%")

# Save data to JSON file with success and failure rates
data_log["summary"] = {
    "success_rate": success_rate,
    "failure_rate": failure_rate}

# Save data to JSON file
with open("iteration_data.json", "w") as json_file:
    json.dump(data_log, json_file, indent=4)

# Plotting
if add_uncertainty_to == "position":
    # Scatter plot for position uncertainty
    sampled_x, sampled_y = zip(*selected_positions)
    plt.figure()
    plt.scatter(sampled_x, sampled_y, c=['green' if success else 'red' for success in grasp_results])
    plt.xlabel("Sampled X Values")
    plt.ylabel("Sampled Y Values")
    #plt.title(" Uncertainty in object position for Normal distribution with standard deviation of 0.01 ")
    plt.title(
        f"Uncertainty in object position for Normal distribution with standard deviation of 0.005\n\nSuccess Rate: {success_rate * 100:.2f}%, Failure Rate: {failure_rate * 100:.2f}%")
    # Create custom legends
    # Create custom legends
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Success')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Fail')
    plt.legend(handles=[green_patch, red_patch])
    plt.show()
elif add_uncertainty_to == "width":
    # Scatter plot for gripper width uncertainty
    plt.figure()
    plt.scatter(iteration_numbers, sampled_widths, c=['green' if success else 'red' for success in grasp_results])
    plt.xlabel("Sample Number")
    plt.ylabel("Sampled Gripper Width")
    plt.title(f"Sampled Gripper Widths with Grasp Results\n\nSuccess Rate: {success_rate * 100:.2f}%, Failure Rate: {failure_rate * 100:.2f}%")
    # Create custom legends
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Success')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Fail')
    plt.legend(handles=[green_patch, red_patch])
    plt.show()
elif add_uncertainty_to == "both":
    # 3D scatter plot for both position and width uncertainty
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sampled_x, sampled_y = zip(*selected_positions)
    ax.scatter(sampled_x, sampled_y, sampled_widths, c=['green' if success else 'red' for success in grasp_results])
    # Create custom legends
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Success')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Fail')
    plt.legend(handles=[green_patch, red_patch])
    ax.set_xlabel("Sampled X values")
    ax.set_ylabel("Sampled Y values")
    ax.set_zlabel("Sampled Gripper Width")
    ax.set_title("Sampled Positions and Gripper Widths with Grasp Results")
    plt.show()