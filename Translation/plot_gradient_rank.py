import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the current working directory
current_path = os.getcwd()
print("Current working directory:", current_path)

# Define the path to your CSV file
csv_file_path = './results/csv/low_rank_info_grank_encoderb0l1wi.csv'  # Replace with the actual path to your CSV file

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Plot the 'rank' against 'step' as a scatter plot (no connecting lines, smaller marker size)
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['rank'], marker='o', linestyle='', markersize=3, color='b', label='Rank')
plt.xlabel('Step')
plt.ylabel('Rank')
plt.title('Rank over Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Define the path to save the plot
plot_save_path = './results/plots/rank_over_steps.png'  # Replace with your desired save path
# Save the plot to the specified path
plt.savefig(plot_save_path, dpi=300)
plt.show()