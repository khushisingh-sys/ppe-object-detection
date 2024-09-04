import argparse
from ultralytics import YOLO
import cv2
import os
from PIL import Image


classes = {
 0:'hard-hat',
 1:'gloves',
 2:'mask',
 3:'glasses',
 4:'boots',
 5:'vest',
 6:'ppe-suit',
 7:'ear-protector',
 8:'safety-harness'
}


def get_files_in_directory(directory_path):
   """
   Gets a list of all files within a specified directory.


   Args:
       directory_path (str): The path to the directory you want to list files from.


   Returns:
       list: A list of filenames within the directory.
   """


   files = []
   for filename in os.listdir(directory_path):
       file_path = os.path.join(directory_path, filename)
       if os.path.isfile(file_path):  # Check if it's a file (not a directory)
           files.append(file_path)


   return files


def save_annotated_image(image_path, person_det_model, ppe_det_model, output_dir):
 person_results = person_det_model(image_path, conf=0.388, classes = [0])
 image = Image.open(image_path)
 basename = os.path.basename(image_path)
 basename_no_ext = os.path.splitext(basename)[0]


 crop_count = 0
 for person_result in person_results:
   boxes = person_result.boxes  # Boxes object for bounding box outputs
   for xyxy in boxes.xyxy.tolist():
     cropped_image = image.crop(xyxy)
     cropped_image_path = os.path.join(output_dir, basename_no_ext +'_unannotated_' + str(crop_count) + '.jpg')
     cropped_image.save(cropped_image_path)
     cropped_image = cv2.imread(cropped_image_path)
     ppe_results = ppe_det_model(cropped_image_path, classes = [0,1,2,3,4,5,6,7,8], conf = 0.418)
     for ppe_result in ppe_results:
       ppe_boxes = ppe_result.boxes  # Boxes object for bounding box outputs
       ppe_xyxy_s = ppe_boxes.xyxy.tolist()
       for cls_ind, ppe_xyxy in zip(ppe_boxes.cls, ppe_xyxy_s):
         start_point = (int(ppe_xyxy[0]), int(ppe_xyxy[1]))  # (x, y) coordinates of the top-left corner
         end_point = (int(ppe_xyxy[2]), int(ppe_xyxy[3]))  # (x, y) coordinates of the bottom-right corner
         cv2.rectangle(cropped_image, start_point, end_point, (255, 0, 0), 2)
         cv2.putText(cropped_image, classes[int(cls_ind)], (int((ppe_xyxy[0] + ppe_xyxy[2])/2), int((ppe_xyxy[1] + ppe_xyxy[3])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 0), 1)
     cropped_image_path = os.path.join(output_dir, basename_no_ext +'annotated' + str(crop_count) + '.jpg')
     cv2.imwrite(cropped_image_path, cropped_image)
     crop_count += 1




def main():
   parser = argparse.ArgumentParser(description="Your program's description")


   # Add arguments
   parser.add_argument("--input_dir", help="Input Dir path")
   parser.add_argument("--output_dir", help="Output Dir path")
   parser.add_argument("--person_det_model", help="Path of Person Detection Model")
   parser.add_argument("--ppe_detection_model", help="Path of PPE Detection Model")
   # Parse the arguments
   args = parser.parse_args()


   person_det_model =  YOLO(args.person_det_model)
   ppe_det_model =  YOLO(args.ppe_detection_model)
   images = get_files_in_directory(args.input_dir)
   if not os.path.exists(args.output_dir):
     os.makedirs(args.output_dir)
   for image_path in images:
     save_annotated_image(image_path, person_det_model, ppe_det_model, args.output_dir)


if __name__ == "__main__":
   main()
