import numpy as np
import argparse
import tensorflow as tf
import cv2
import pygame
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import serial
import time

# Patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model.signatures['serving_default']

def run_inference_for_single_image(detect_fn, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = detect_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def run_inference(detect_fn, category_index, cap):
    try:
        SerialObj = serial.Serial('COM6', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return

    frame_delay = 1.0  # Delay between frames in seconds

    while True:
        ret, image_np = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        output_dict = run_inference_for_single_image(detect_fn, image_np)

        if output_dict['num_detections'] > 0:
            for i in range(output_dict['num_detections']):
                class_id = output_dict['detection_classes'][i]
                score = output_dict['detection_scores'][i]
                if score > 0.5 and category_index[class_id]['name'] != "Chips":
                    class_name = category_index[class_id]['name']
                    print(f"Object: {class_name}, Score: {score:.2f}")
                    try:
                        # Clear the serial input buffer
                        SerialObj.reset_input_buffer()

                        if class_name == "WaterBottle":
                            play_sound(r'C:\Users\camel\OneDrive\Desktop\GarbageCollector\models\research\object_detection\Recording\water\one_in_three.mp3')
                            print('playing sound using pygame')
                            SerialObj.write(b'B')
                        elif class_name == "JuiceBox":
                            play_sound(r'C:\Users\camel\OneDrive\Desktop\GarbageCollector\models\research\object_detection\Recording\Juice\decorate.mp3')
                            print('playing sound using pygame')
                            SerialObj.write(b'J')
                        elif class_name == "Tissue":
                            play_sound(r'C:\Users\camel\OneDrive\Desktop\GarbageCollector\models\research\object_detection\Recording\Tissue\untitled.mp3')
                            print('playing sound using pygame')
                            SerialObj.write(b'N')
                        elif class_name == "Plate":
                            play_sound(r'C:\Users\camel\OneDrive\Desktop\GarbageCollector\models\research\object_detection\Recording\Plate\wood.mp3')
                            print('playing sound using pygame')
                            SerialObj.write(b'P')
                        elif class_name == "Chips":
                            SerialObj.write(b'C')
                        elif class_name == "Wrapper":
                            SerialObj.write(b'W')

                        time.sleep(2)  # Wait for the Arduino to process the command
                        break  # Break after the first detection to avoid backlogs

                    except serial.SerialException as e:
                        print(f"Error writing to serial port: {e}")

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        time.sleep(frame_delay)  # Delay to reduce frame rate

    SerialObj.close()
    print("Serial port closed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera device index')
    args = parser.parse_args()

    detect_fn = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    cap = cv2.VideoCapture(args.camera)
    run_inference(detect_fn, category_index, cap)
