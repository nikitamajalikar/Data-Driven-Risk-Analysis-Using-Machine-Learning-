import rospy
from geometry_msgs.msg import Pose
import numpy as np
from hw_interface import HwInterface  # Assume the HwInterface class is in a file named hw_interface.py
#from logger import Logger  # Ensure the Logger class is in the same directory or properly imported
import matplotlib.pyplot as plt

import json
from scipy.stats import alpha, beta, lognorm
from envs.Locate import Locate
from envs.panda_transformer import PandaTransformer

# Initialize the logger
#logger = Logger()

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

# Print object position
#Object_position = hw_interface.get_sim_state()
#print("Object position:", Object_position)

my_trans = PandaTransformer()
my_trans.init_pose()

locate = Locate()
base_coordinates, relative_position = locate.locate_colored_object(colormode='cube')
print("Object position:", base_coordinates)

# Function to sample new x, y coordinates based on a specified distribution
def sample_new_position(x, y, distribution="alpha", num_samples=50, **kwargs):
    if distribution == "normal":
        std_dev = kwargs.get("std_dev", 0.005)
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
        low = kwargs.get("low", x - 0.001)
        high = kwargs.get("high", x + 0.001)
        sampled_x = np.random.uniform(low, high, num_samples)
        low = kwargs.get("low", y - 0.001)
        high = kwargs.get("high", y + 0.001)
        sampled_y = np.random.uniform(low, high, num_samples)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return sampled_x, sampled_y


# Choose distribution and sample new positions for x and y
distribution_type = "alpha"  # Change this to use a different distribution
sampled_x, sampled_y = sample_new_position(base_coordinates[0], base_coordinates[1], distribution=distribution_type)
print("Sampled position:", sampled_x, sampled_y)

# Plotting the sampled positions
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(sampled_x, bins=30, density=True, alpha=0.6, color='g')
plt.title("Histogram of 50 samples for x")
plt.xlabel("x-coordinate values")
plt.ylabel("Density")

plt.subplot(1, 2, 2)
plt.hist(sampled_y, bins=30, density=True, alpha=0.6, color='b')
plt.title("Histogram of 50 samples for y")
plt.xlabel("y-coordinate values")
plt.ylabel("Density")

plt.show()

## List to store data
data_log = {
   "initial_positions": {
       "robot_position": robot_position,
 #      "object_position": Object_position.tolist() if isinstance(Object_position, np.ndarray) else Object_position,
       "gripper_position": gripper_position,
   },
   "iterations": []
}
## List to store data
iteration_numbers = []
selected_positions = []
grasp_results = []


for i in range(50):
    # Select a different sample for each iteration
    index = i  # Use the loop index to select a different sample each time
    selected_x = float(sampled_x[index])
    selected_y = float(sampled_y[index])

    print(f"Iteration: {i + 1}, Selected position for action: x={selected_x}, y={selected_y}")

    iteration_data = {
        "iteration": i + 1,
        "selected_position": {"x": selected_x, "y": selected_y},
        "actions": []
    }

    # Open the gripper
    hw_interface.move_gripper()
    iteration_data["actions"].append({"action": "move_gripper", "status": "opened"})

    # Test move_cartesian method
    success = hw_interface.move_cartesian(x=selected_x, y=selected_y, z=0.2, o_x=1, o_y=0, o_z=0, o_w=0, vel=0.1,
                                          acc=0.1)
    #iteration_data["actions"].append({"action": "move_cartesian", "status": "Success" if success else "Failed"})
    print("Move Cartesian:", "Success" if success else "Failed")
    hw_interface.move_cartesian(z=0.02, vel=0.1, acc=0.1)
    #iteration_data["actions"].append({"action": "move_cartesian", "status": "moved down"})

    # Test grasp_object method
    success = hw_interface.grasp_object()
    iteration_data["actions"].append(
       {"action": "grasp_object", "status": "Success" if success else "No object grasped"})
    print("Grasping object:", "Success" if success else "No object grasped")

    # Test place object in another coordinate
    hw_interface.move_cartesian(z=0.2, vel=0.1, acc=0.1)
    hw_interface.move_cartesian(x=base_coordinates[0], y=base_coordinates[1], vel=0.1, acc=0.1)
    hw_interface.move_cartesian(z=0.02, vel=0.1, acc=0.1)
    hw_interface.move_gripper()
    hw_interface.move_cartesian(z=0.2, vel=0.1, acc=0.1)
    hw_interface.move_cartesian(x=0.2, y=0.1, vel=0.1, acc=0.1)

    # Append iteration data to data log
    data_log["iterations"].append(iteration_data)

    #collect data for plotting
    iteration_numbers.append(i + 1)
    selected_positions.append((selected_x, selected_y))
    grasp_results.append(success)

# Convert NumPy arrays to lists for JSON serialization
def convert_to_serializable(obj):
  if isinstance(obj, np.ndarray):
      return obj.tolist()
        #return obj

## Save the log data to a JSON file
with open("hw_interface_log.json", "w") as f:
   json.dump(data_log, f, indent=4, default=convert_to_serializable)

#plotting bar graph after execution
colors = ['g' if result else 'r' for result in grasp_results]

plt.figure(figsize=(14, 7))

# Bar graph for x positions
plt.subplot(1, 2, 1)
bars_x = plt.bar(iteration_numbers, [pos[0] for pos in selected_positions], color=colors)
plt.title('Selected X Positions for Each Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Selected X Position')

# Bar graph for y positions
plt.subplot(1, 2, 2)
bars_y = plt.bar(iteration_numbers, [pos[1] for pos in selected_positions], color=colors)
plt.title('Selected Y Positions for Each Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Selected Y Position')

# Create custom legends
green_patch = plt.Line2D([0], [0], color='g', lw=4, label='Success')
red_patch = plt.Line2D([0], [0], color='r', lw=4, label='Fail')
plt.legend(handles=[green_patch, red_patch])
plt.tight_layout()
plt.show()

