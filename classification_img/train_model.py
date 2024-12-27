from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import pickle

DATASET_PATH = f'{os.path.abspath(os.path.dirname(__file__))}/training/train_data'
SAVE_PATH_MODEL = f'{os.path.abspath(os.path.dirname(__file__))}/model_save/tf_model.keras'
SAVE_PATH_DICT = f'{os.path.abspath(os.path.dirname(__file__))}/model_save/tf_labels.pkl'

def setup():
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalizing pixel values to [0, 1]
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,  # Path to your dataset
        target_size=(512, 512),  # Resize images to 224x224 (adjust as needed)
        batch_size=32,  # Set batch size
        class_mode='categorical'  # Since it's multi-class classification
    )
    return train_generator


def do_train(train_generator):
    model=Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid'))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid'))

    # Flatten the feature maps
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Prevent overfitting
    model.add(Dense(train_generator.num_classes, activation='softmax'))  # Output layer for multi-class
    
    return model

def train():
    train_generator = setup()
    model = do_train(train_generator)
    
    # Summary of the model
    print(model.summary())

    model.compile(
        optimizer='adam',  # Adaptive moment estimation optimizer
        loss='categorical_crossentropy',  # For multi-class classification
        metrics=['accuracy']  # Track accuracy during training
    )

    # Save the model
    model.save(SAVE_PATH_MODEL)

    # Save label dict
    class_labels = train_generator.class_indices  # This will give a dictionary like {'class1': 0, 'class2': 1, ...}
    # Reverse the dictionary to map index to label
    label_map = {v: k for k, v in class_labels.items()}
    with open(SAVE_PATH_DICT, 'wb') as f:
        pickle.dump(label_map, f)


if __name__== "__main__":
    train()