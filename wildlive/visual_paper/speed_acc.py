import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
methods = ['1 ', '2', '4', '8', '16', '24']
jetson_speed = [7.53, 6.94, 5.87, 4.59, 3.28, 2.45]  # Speed values 1
tesla_speed = [6.02, 4.98, 4.17, 3.30, 2.52, 2.24]  # Speed values 2
accuracy = [73.73, 75.35, 75.08, 74.60, 74.55, 74.71]  # Accuracy values
nation = [0.67, 0.71, 0.71]  # nation speed information
acc_points = [62.43, 50.29, 59.06]  # additional accuracy points

# Create the figure and axes objects
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot jetson_speed and tesla_speed on the left y-axis
color1 = 'tab:blue'
color2 = 'tab:green'
ax1.set_xlabel('Detection Window',fontsize=20)
plt.xticks(fontsize=12) # set x axis label font size

ax1.set_ylabel('Speed (fps)', color=color1,fontsize=20)
ax1.plot(methods, jetson_speed, marker='o', color=color1, label='WildLive speed on Jetson',linewidth=5)
ax1.plot(methods, tesla_speed, marker='^', color=color2, label='WildLive speed on Tesla')
ax1.tick_params(axis='y', labelcolor=color1,labelsize=15)
ax1.legend(loc='upper left')

# Plot nation speed markers on the left y-axis with specific markers
ax1.plot(np.repeat(methods[0], 1), nation[0:1], marker='D', markersize=8, color='purple', linestyle='none', label='ByteTrack speed on Tesla')
ax1.plot(np.repeat(methods[0], 1), nation[1:2], marker='x', markersize=8, color='black', linestyle='none', label='OC-SORT speed on Tesla')
ax1.plot(np.repeat(methods[0], 1), nation[2:3], marker='*', markersize=8, color='goldenrod', linestyle='none', label='SORT speed on Tesla')
ax1.legend(loc='upper left',fontsize=15)

# Set the y-axis limits for speed
ax1.set_ylim(0, 12)
ax1.grid(True, linestyle='--', alpha=0.5) #add grid

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.set_ylim(20, 120)
ax2.grid(True, linestyle='--', alpha=0.5) #add grid

# Plot accuracy on the right y-axis
color3 = 'tab:red'
ax2.set_ylabel('Accuracy (%)', color=color3,fontsize=20)
ax2.plot(methods, accuracy, marker='s', color=color3, label='WildLive')
ax2.tick_params(axis='y', labelcolor=color3,labelsize=15)
ax2.legend(loc='upper right',fontsize=20)

# Plot additional accuracy points on the right y-axis with specific markers
ax2.plot(np.repeat(methods[-1], 1), acc_points[0:1], marker='D', markersize=8, color='purple', linestyle='none', label='ByteTrack')
ax2.plot(np.repeat(methods[-1], 1), acc_points[1:2], marker='x', markersize=8, color='black', linestyle='none', label='OC-SORT')
ax2.plot(np.repeat(methods[-1], 1), acc_points[2:3], marker='*', markersize=8, color='goldenrod', linestyle='none', label='SORT')
ax2.legend(loc='upper right',fontsize=15)

# Add title and adjust layout
#plt.title('Speed vs. Accuracy Trade-off')
fig.tight_layout()

# Show the plot
plt.show()
