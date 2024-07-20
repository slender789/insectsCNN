import os
import re
import time
from preprocess_raw_image import countInsects
from use_weights_cnn import predict_insect

def extract_number_from_filename(filename):
    # Use regex to find the number between the last underscore and .JPG
    match = re.search(r'_(\d+)\.JPG$', filename)
    if match:
        return match.group(1)
    return None

def query_images_in_folder(folder_path):
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG"):
            number = extract_number_from_filename(filename)
            if number is not None:
                results[filename] = number
    return results

def main():
    folder_path = 'DENDROCTONUS_MEXICANUS'  # Replace with the path to your folder
    results = query_images_in_folder(folder_path)
    
    with open('detection_results_segmentation.txt', 'w') as file:
        for filename, realInsects in results.items():   
            model_weights_path = 'insect_detection_model.weights.h5'
            
            start_count_time = time.time()
            candidate_insects_images = countInsects(f'{folder_path}/{filename}')
            end_count_time = time.time()
            total_count_time = end_count_time - start_count_time
    
            insects = 0
            notInsects = 0
            
            total_predict_time = 0
            for candidate_insect in candidate_insects_images:
                start_predict_time = time.time()
                isInsect = predict_insect(model_weights_path, fromPath=False, image=candidate_insect)
                end_predict_time = time.time()
                
                predict_time = end_predict_time - start_predict_time
                total_predict_time += predict_time

                if isInsect:
                    insects += 1
                else:
                    notInsects += 1
            
            file.write(f'Filename: {filename}, Detected Insects: {insects} vs. Real Insects: {realInsects}\n')
            file.write(f'Not insects: {notInsects}\n')
            file.write(f'Total time taken to count insects: {total_count_time:.2f} seconds\n')
            file.write(f'Total time taken to predict insects: {total_predict_time:.2f} seconds\n\n')

if __name__ == "__main__":
    main()
