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
    Calculate performance metrics: exact matches, overcounts, undercounts,
    mean absolute error, and mean absolute percentage error (MAPE).
    """
    # Calculate absolute errors
    absolute_errors = np.abs(y_pred - y_truth)
    mean_absolute_error = np.mean(absolute_errors)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_pred - y_truth) / y_truth)) * 100 if np.all(y_truth != 0) else np.nan

    # Count the number of exact matches, overcounts, and undercounts
    exact_matches = np.sum(y_pred == y_truth)
    overcounts = np.sum(y_pred > y_truth)
    undercounts = np.sum(y_pred < y_truth)

    return {
        "exact_matches": exact_matches,
        "overcounts": overcounts,
        "undercounts": undercounts,
        "mean_absolute_error": mean_absolute_error,
        "mape": mape,
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
    print(f"Exact matches: {metrics['exact_matches']}")
    print(f"Overcounts: {metrics['overcounts']}")
    print(f"Undercounts: {metrics['undercounts']}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")

    # Plot the distribution of absolute errors
    plot_error_distribution(metrics)

if __name__ == "__main__":
    main()