import os
import csv
import numpy as np
import cv2
import cv2 as cv

def flip_image(image):
    image = cv2.flip(image, 1)
    image = cv2.flip(image, 0)
    return image

def rotated_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image
def main():
    input_path = 'data/asl_alphabet_train/asl_alphabet_train'
    dirs = os.listdir(input_path)
    output_path = 'data/asl_alphabet_train/asl_alphabet_train_rotated_v3'
    index = 0
    min_angle = -45
    max_angle = 45
    list_data = ['L', 'Z', 'B', 'S', 'V']
    for dir in dirs:
        if dir not in list_data: continue
        folder_output = os.path.join(output_path, dir)
        os.makedirs(folder_output, exist_ok= True)

        index = 0
        fold = os.path.join(input_path, dir)
        imgs = os.listdir(fold)
        for img in imgs:
            index = index + 1
            img_path = os.path.join(fold, img)
            image = cv.imread(img_path)

            # Flip image
           # image = flip_image(image)

            # Generate a random rotation angle within the specified range
            angle = np.random.randint(min_angle, max_angle + 1)

            # Rotated image
            rotated_img = rotated_image(image, angle)

            # Save image
            cv.imwrite(os.path.join(folder_output, img), rotated_img)
            print("Fold: %s, img: %s, index: %s" %(fold, img, index))
            # cv.imshow('L_1_%s' % index, image)
            # if(index == 3):
            #     break
        # cv.waitKey(0)

    # Specify the output directory where rotated images will be saved
    # output_dir = 'path/to/your/output/folder/'

    # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)


    # print(f"Rotated images saved to: {output_dir}")


if __name__ == '__main__':
    main()