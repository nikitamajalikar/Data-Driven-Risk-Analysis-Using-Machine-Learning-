import rospy
from geometry_msgs.msg import Pose
import numpy as np
from hw_interface import HwInterface  # Assume the HwInterface class is in a file named hw_interface.py
import matplotlib.pyplot as plt
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
#my_trans.init_pose()

locate = Locate()
base_coordinates, relative_position = locate.locate_colored_object(colormode='cube')

# Print object position
#object_position = hw_interface.get_sim_state()
#print("Object position:", object_position)

# Function to sample new gripper widths based on a specified distribution
def sample_gripper_widths(mean_width, distribution="normal", num_samples=50, **kwargs):
    if distribution == "normal":
        std_dev = kwargs.get("std_dev", 0.005)
        sampled_widths = np.random.normal(mean_width, std_dev, num_samples)
    elif distribution == "triangular":
        left = kwargs.get("left", mean_width - 0.005)
        mode = kwargs.get("mode", mean_width)
        right = kwargs.get("right", mean_width + 0.005)
        sampled_widths = np.random.triangular(left, mode, right, num_samples)
    #elif distribution == "alpha":
        #a = kwargs.get("a", 2)
       #sampled_widths = alpha.rvs(a, loc=mean_width, scale=0.001, size=num_samples)
    elif distribution == "beta":
        a = kwargs.get("a", 2)
        b = kwargs.get("b", 0.5)
        sampled_widths = beta.rvs(a, b, loc=mean_width, scale=0.0045, size=num_samples)
    elif distribution == "lognormal":
        mean = kwargs.get("mean", 0)
        sigma = kwargs.get("sigma", 0.0001)
        sampled_widths = lognorm.rvs(sigma, loc=mean_width, scale=np.exp(mean), size=num_samples)
    elif distribution == "uniform":
        sampled_widths = np.random.uniform(mean_width - 0.001, mean_width + 0.001, num_samples)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return sampled_widths

# Set mean gripper width and sample new widths
mean_gripper_width = 0.03 # Assuming the nominal width for grasping is 0.03 meters
distribution_type = "normal"  # Change this to use a different distribution
sampled_gripper_widths = sample_gripper_widths(mean_gripper_width, distribution=distribution_type)
print("Sampled gripper widths:", sampled_gripper_widths)

# Plotting the sampled gripper widths
plt.figure(figsize=(10, 5))
plt.hist(sampled_gripper_widths, bins=30, density=True, alpha=0.6)
plt.title(f"Histogram of 50 samples for gripper width ({distribution_type} distribution)")
plt.xlabel("Gripper width values")
plt.ylabel("Density")
plt.show()

# Convert initial positions to lists if they are numpy arrays
initial_robot_position = robot_position.tolist() if isinstance(robot_position, np.ndarray) else robot_position
#initial_object_position = object_position.tolist() if isinstance(object_position, np.ndarray) else object_position
initial_gripper_position = gripper_position.tolist() if isinstance(gripper_position, np.ndarray) else gripper_position

# List to store data
data_log = {

    "initial_positions": {

        "robot_position": initial_robot_position,

        #"object_position": initial_object_position,

        "gripper_position": initial_gripper_position,

    },

    "iterations": []

}
# Lists to store data for plotting
iteration_numbers = []
sampled_widths = []
grasp_results = []


for i in range(50):
    # Select a different sample for each iteration
    sampled_gripper_width = sampled_gripper_widths[i]

    print(f"Iteration: {i + 1}, Sampled gripper width for action: {sampled_gripper_width}")

    iteration_data = {
        "iteration": i + 1,
        "sampled_gripper_width": sampled_gripper_width,
        "actions": []
    }

    # Open the gripper
    hw_interface.move_gripper()
    iteration_data["actions"].append({"action": "move_gripper", "status": "opened"})
    # Move the gripper to the sampled width
    #hw_interface.move_gripper(width=sampled_gripper_width)
    #iteration_data["actions"].append({"action": "move_gripper", "status": "moved to sampled width"})

    # Move the gripper to a position above the object
    success = hw_interface.move_cartesian(x=base_coordinates[0], y=base_coordinates[1], z=0.2, o_x=1, o_y=0, o_z=0, o_w=0, vel=0.1, acc=0.1)
    #iteration_data["actions"].append({"action": "move_cartesian", "status": "Success" if success else "Failed"})
    print("Move Cartesian:", "Success" if success else "Failed")
    hw_interface.move_cartesian(z=0.02, vel=0.1, acc=0.1)
    #iteration_data["actions"].append({"action": "move_cartesian", "status": "moved down"})

    # Test grasp_object method
    success = hw_interface.grasp_object(width=sampled_gripper_width)
    iteration_data["actions"].append({"action": "grasp_object", "status": "Success" if success else "Failed"})
    print("Grasping object:", "Success" if success else "Failed")

    # Move the gripper up after grasping
    hw_interface.move_cartesian(z=0.2, vel=0.1, acc=0.1)
   # iteration_data["actions"].append({"action": "move_cartesian", "status": "moved up after grasp"})
    hw_interface.move_cartesian(x=base_coordinates[0], y=base_coordinates[1], vel=0.1, acc=0.1)
    hw_interface.move_cartesian(z=0.02, vel=0.1, acc=0.1)
    hw_interface.move_gripper()
    hw_interface.move_cartesian(z=0.2, vel=0.1, acc=0.1)
    hw_interface.move_cartesian(x=0.2, y=0.1, vel=0.1, acc=0.1)

    # Log iteration data
    data_log["iterations"].append(iteration_data)

    # Collect data for plotting
    iteration_numbers.append(i + 1)
    sampled_widths.append(sampled_gripper_width)
    grasp_results.append(success)

# Save the log to a file
with open("data_log.json", "w") as f:
   json.dump(data_log, f, indent=4)


# Plot scatter plot of sampled gripper widths with color coding for success and failure
colors = ['g' if result else 'r' for result in grasp_results]

plt.figure(figsize=(12, 6))
plt.scatter(sampled_widths, iteration_numbers, color=colors)

# Customize the plot
plt.title('Gripper Widths for Each Iteration with Success and Failure')
plt.xlabel('Sampled Gripper Width (meters)')
plt.ylabel('Sample Number')
plt.grid(True)

# Create custom legends
green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Success')
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Fail')
plt.legend(handles=[green_patch, red_patch])

# Show the plot
plt.show()