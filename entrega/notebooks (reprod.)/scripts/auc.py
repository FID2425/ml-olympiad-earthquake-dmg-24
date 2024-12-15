import pandas as pd
from imcp import plot_mcp_curve, imcp_score
import matplotlib.pyplot as plt

def calculate_and_plot_auc(original_data_path, predicted_data_path, output_plot_path):
    # Load the data
    original_data = pd.read_csv(original_data_path)
    predicted_data = pd.read_csv(predicted_data_path)

    # Extract y_true and y_score
    y_true = original_data["damage_grade"]
    y_score = predicted_data.drop(["damage_grade_pred"], axis=1)

    # Plot the IMCP curve
    plot_mcp_curve(y_true, y_score)
    plt.savefig(output_plot_path)  # Save the plot

    # Calculate the area under the IMCP curve
    area = imcp_score(y_true, y_score, abs_tolerance=0.0000001)

    return area

if __name__ == "__main__":
    import sys
    original_data_path = sys.argv[1]  # Path to the original data CSV
    predicted_data_path = sys.argv[2]  # Path to the predicted data CSV
    output_plot_path = sys.argv[3]  # Path to save the plot

    area = calculate_and_plot_auc(original_data_path, predicted_data_path, output_plot_path)
    print(f"Area under IMCP curve: {area:.4f}")