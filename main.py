import os
from preprocess_raw_image import countInsects
from use_weights import predict_insect

if __name__ == "__main__":
    image_path = 'IMG_5790_raw.JPG'  # Replace with the path to your folder containing images
    model_weights_path = 'insect_detection_model.weights.h5'

    candidate_insects_images = countInsects(image_path)
    
    insects = 0
    notInsects = 0
    
    for candidate_insect in candidate_insects_images:
        isInsect = predict_insect(model_weights_path, fromPath=False, image=candidate_insect)
        if isInsect:
            insects += 1
        else:
            notInsects += 1
    
    print(f'Insects: {insects}')
    print(f'Not insects: {notInsects}')
