from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a new model from YAML

    # Train the model with 1 GPU
    results = model.train(data='config.yaml', epochs=100)

if __name__ == '__main__':
    main()