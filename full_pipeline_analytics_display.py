import csv
import numpy as np
import matplotlib.pyplot as plt

def read_results(results_file):
    """
    Read the CSV file containing the results of the insect counting.
    Returns filenames, y_truth, y_pred, and elapsed_times as numpy arrays.
    """
    filenames = []
    y_truth = []
    y_pred = []
    elapsed_times = []

    with open(results_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            filenames.append(row[0])
            y_truth.append(int(row[1]))
            y_pred.append(int(row[2]))
            elapsed_times.append(float(row[3]))

    return filenames, np.array(y_truth), np.array(y_pred), np.array(elapsed_times)

def calculate_metrics(y_truth, y_pred):
    """
    Calculate performance metrics: overcounts, undercounts,
    mean absolute error, mean absolute percentage error (MAPE),
    and average relative accuracy.
    """
    absolute_errors = np.abs(y_pred - y_truth)
    mean_absolute_error = np.mean(absolute_errors)

    mape = np.mean(np.abs((y_pred - y_truth) / y_truth)) * 100 if np.all(y_truth != 0) else np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_accuracy = 1 - (absolute_errors / y_truth)
        relative_accuracy = np.where(y_truth == 0, np.nan, relative_accuracy)
        avg_relative_accuracy = np.nanmean(relative_accuracy)

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
    plt.figure(figsize=(8, 6))
    plt.hist(metrics['absolute_errors'], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribución de error absoluto', fontsize=14)
    plt.xlabel('Error Absoluto', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig("docs/error_distribution.png")
    plt.close()

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
    plt.savefig("docs/error_distribution.png")

def plot_time_distribution(elapsed_times):
    plt.figure(figsize=(8, 6))  # Create a fresh figure
    plt.hist(elapsed_times, bins=20, color='lightblue', edgecolor='black')
    plt.title('Distribución del tiempo de procesamiento por imagen', fontsize=14)
    plt.xlabel('Tiempo (segundos)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig("docs/time_distribution.png")
    plt.close()  # Close the figure to prevent reuse

    print("\nElapsed Time Summary:")
    print(f"- Average Time per Image: {np.mean(elapsed_times):.2f} s")
    print(f"- Min Time: {np.min(elapsed_times):.2f} s")
    print(f"- Max Time: {np.max(elapsed_times):.2f} s")

def main(results_file="docs/whole_pipeline_results.csv"):
    # Read results
    _, y_truth, y_pred, elapsed_times = read_results(results_file)

    # Calculate metrics
    metrics = calculate_metrics(y_truth, y_pred)

    # Display performance metrics
    print("Performance Metrics:")
    print(f"Overcounts: {metrics['overcounts']}")
    print(f"Undercounts: {metrics['undercounts']}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    print(f"Average Relative Accuracy: {metrics['avg_relative_accuracy']:.2%}")

    # Plot metrics
    plot_error_distribution(metrics)
    plot_time_distribution(elapsed_times)

if __name__ == "__main__":
    main()
