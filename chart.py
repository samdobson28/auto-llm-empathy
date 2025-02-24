import matplotlib.pyplot as plt
import pandas as pd

# Data from your analysis:
methods = ["Transcript Rating", "Audio Features Rating", "Combined Rating"]
mae = [1.87, 2.00, 1.99]          # Mean Absolute Error
accuracy = [37.5, 0, 17]           # Accuracy in percentage

# Create a figure with two subplots: one for MAE and one for Accuracy
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart for MAE
axes[0].bar(methods, mae, color='skyblue')
axes[0].set_title("Mean Absolute Error")
axes[0].set_ylabel("MAE")
axes[0].set_ylim(0, max(mae)+1)
for i, v in enumerate(mae):
    axes[0].text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

# Bar chart for Accuracy
axes[1].bar(methods, accuracy, color='salmon')
axes[1].set_title("Accuracy (%)")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 100)
for i, v in enumerate(accuracy):
    axes[1].text(i, v + 2, f"{v}%", ha='center', fontweight='bold')

plt.suptitle("Performance Comparison: Transcript vs Audio Features vs Combined", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure as an image
plt.savefig("performance_comparison.png", dpi=300)
plt.show()

# Additionally, create a concise table using pandas and print it:
data = {
    "Method": methods,
    "Mean Absolute Error": mae,
    "Accuracy (%)": accuracy
}
summary_table = pd.DataFrame(data)
print(summary_table)
