import csv
import numpy as np
import matplotlib.pyplot as plt

def read_results(results_file):
    """
    Read the CSV file containing the results of the insect counting.
    Returns lists of filenames, ground truth values (y_truth), and predicted values (y_pred).
    """
    filenames = []
    y_truth = []
    y_pred = []

    with open(results_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            filenames.append(row[0])
            y_truth.append(int(row[1]))
            y_pred.append(int(row[2]))

    return filenames, np.array(y_truth), np.array(y_pred)

def calculate_metrics(y_truth, y_pred):
    """
    Calculate performance metrics: overcounts, undercounts,
    mean absolute error, mean absolute percentage error (MAPE),
    and average relative accuracy.
    """
    absolute_errors = np.abs(y_pred - y_truth)
    mean_absolute_error = np.mean(absolute_errors)

    # MAPE
    mape = np.mean(np.abs((y_pred - y_truth) / y_truth)) * 100 if np.all(y_truth != 0) else np.nan

    # Relative accuracy per image, skipping zero ground truths
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_accuracy = 1 - (absolute_errors / y_truth)
        relative_accuracy = np.where(y_truth == 0, np.nan, relative_accuracy)
        avg_relative_accuracy = np.nanmean(relative_accuracy)

    # Count over/under
    overcounts = np.sum(y_pred > y_truth)
    undercounts = np.sum(y_pred < y_truth)

    return {
        "overcounts": overcounts,
        "undercounts": undercounts,
        "mean_absolute_error": mean_absolute_error,
        "mape": mape,
        "avg_relative_accuracy": avg_relative_accuracy,
        "absolute_errors": absolute_errors
    }

def plot_error_distribution(metrics):
    """
    Plot the distribution of absolute errors.
    """
    plt.hist(metrics['absolute_errors'], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribución de error absoluto', fontsize=14)
    plt.xlabel('Error Absoluto', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

def main(results_file="docs/segmentation_results.csv"):
    # Read results
    _, y_truth, y_pred = read_results(results_file)

    # Calculate metrics
    metrics = calculate_metrics(y_truth, y_pred)

    # Display the metrics
    print("Performance Metrics:")
    print(f"Overcounts: {metrics['overcounts']}")
    print(f"Undercounts: {metrics['undercounts']}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    print(f"Average Relative Accuracy: {metrics['avg_relative_accuracy']:.2%}")

    # Plot the distribution of absolute errors
    plot_error_distribution(metrics)

if __name__ == "__main__":
    main()
