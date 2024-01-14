import cv2
import os

# Specify this class mapping everytime a new object detection task needs to be performed. Keys: class to be detected, values: class label to be used by Yolo
class_mapping = {'Car': 0}

# Always set these paths before generating new text file labels for Yolo.
folder_path_text_input = r'C:\RWTH Aachen University\Projects\Object Detection\OIDv4_ToolKit\OID\Dataset\train\Car\Label'               # Text files that would be used to generate new labels.
folder_path_images_input = r'C:\RWTH Aachen University\Projects\Object Detection\OIDv4_ToolKit\OID\Dataset\train\Car'                   # Corresponding images folder that have same images as the corresponding text files
folder_path_new_text_output = r'C:\RWTH Aachen University\Projects\Object Detection\OIDv4_ToolKit\OID\Dataset\train\Car\Yolo_labels'    # Folder where new text files need to be stored

if not os.path.exists(folder_path_new_text_output):
    os.makedirs(folder_path_new_text_output)

# Function to give full path to a given .txt file
def give_txt_file_path(txt_file_name):
    return os.path.join(folder_path_text_input, txt_file_name)

# Function to give full output path to a given .txt file
def give_output_txt_file_path(txt_file_name):
    return os.path.join(folder_path_new_text_output, txt_file_name)

# Function to give full path to a given .jpg file
def give_jpg_file_path(txt_file_name):
    name_without_extension = txt_file_name.split(".")[0]
    return os.path.join(folder_path_images_input, name_without_extension + '.jpg')

# Function that takes content: from text files, image_file_path: corresponding image path. It performs some modifications on the content and returns a list of lists that would be written in the new text file
def modify_content(content, image_file_path):
    im = cv2.imread(image_file_path)
    complete_list_of_lists = []
    for line in content:
        line = line.split()
        class_id = class_mapping.get(line[0])
        contents_iterator = map(float, line[1:])
        left, top, right, bottom = contents_iterator
        x_centre = 0.5*(right - left) + left
        y_centre = 0.5*(bottom - top) + top
        width_bb = right-left
        height_bb = bottom-top
        height, width = im.shape[:2]
        x_centre /= width
        width_bb /= width
        y_centre /= height
        height_bb /= height
        modified_line = [class_id, round(x_centre, 5), round(y_centre, 5), round(width_bb, 5), round(height_bb, 5)]
        complete_list_of_lists += [modified_line]
    return complete_list_of_lists

# x = 1             # Only for testing
for file_name in os.listdir(folder_path_text_input):
    if file_name.endswith(".txt"):
        input_file_path = give_txt_file_path(file_name)             # Points to file in Text_Files folder
        output_file_path = give_output_txt_file_path(file_name)     # Points to file in Yolo_labels folder
        input_image_path = give_jpg_file_path(file_name)            # Points to the image in Images folder
        # print(input_image_path)

        # Read the content from the input file
        with open(input_file_path, 'r') as input_file:
            content = input_file.readlines()
            new_content = modify_content(content, input_image_path)
            content_to_write = '\n'.join([' '.join(map(str,sublist)) for sublist in new_content])
            # Below 3 three lines are for testing
            # print(x)
            # print(content_to_write)
            # x += 1

        with open(output_file_path, 'w') as output_file:
            output_file.write(content_to_write)