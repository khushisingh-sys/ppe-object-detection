import argparse
import glob
import os
import xml.etree.ElementTree as ET

def get_classes(class_file_path):
    classes = []
    with open(class_file_path, "r") as file:
        for line in file:
            classes.append(line.split("\n")[0])
    return classes

def get_label_files_in_dir(dir_path):
    labels_list = []
    for filename in glob.glob(dir_path + '/*.xml'):
        labels_list.append(filename)
    return labels_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(label_file_path, output_dir_path, classes):
    basename = os.path.basename(label_file_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(label_file_path)
    out_file = open(os.path.join(output_dir_path, basename_no_ext + '.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()



def main():
    parser = argparse.ArgumentParser(description="Covert Pascal VOC annotations to yolo format")
    parser.add_argument("--base_input_directory_path", type=str, required=True, help="Path of Base input directory")
    parser.add_argument("--output_directory", type=str, required=True, help="Path of Output Directory")
    args = parser.parse_args()
    base_dir = args.base_input_directory_path
    output_dir = args.output_directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_files = get_label_files_in_dir(os.path.join(base_dir, 'labels'))
    classes = get_classes(os.path.join(base_dir, 'classes.txt'))
    print(classes)
    for label_file_path in label_files:
        convert_annotation(label_file_path, output_dir, classes)
    

if __name__ == "__main__":
    main()
