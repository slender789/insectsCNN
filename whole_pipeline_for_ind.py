import time
from preprocess_raw_image import countInsects
from use_weights_cnn import predict_insect

if __name__ == "__main__":
    image_path = 'IMG_5790_raw.JPG'  # Replace with the path to your folder containing images
    model_weights_path = 'insect_detection_model.weights.h5'
    
    start_count_time = time.time()
    candidate_insects_images = countInsects(image_path)
    end_count_time = time.time()
    total_count_time = end_count_time - start_count_time
    
    insects = 0
    notInsects = 0
    
    # Timing the prediction process
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
    
    print(f'Insects: {insects}')
    print(f'Not insects: {notInsects}')
    print(f'Total time taken to count insects: {total_count_time:.2f} seconds')
    print(f'Total time taken to predict insects: {total_predict_time:.2f} seconds')
