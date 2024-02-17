# Face-Mask-Detector

A face mask detector for verifying that a person is wearing a face mask correctly (over mouth and nose). Has two categories: “with mask” and “without mask”. Classification is done with a convolutional neural network, made with Tensorflow Keras. Demo GUI uses OpenCV and the HaarCascade Frontal Face model to apply the face mask model to camera input. Features a data augmentation engine and MatPlotLib training visualization.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)

## Installation

To run this project, you'll need to set up a Conda environment using the provided `face_mask_detector_env.yml` file.

1. **Clone the repository:**

    ```bash
    $ git clone https://github.com/siddhp1/Face-Mask-Detector.git
    ```

2. **Navigate to the project directory:**

    ```bash
    $ cd repository
    ```

3. **Create the Conda environment:**

    ```bash
    $ conda env create -f face_mask_detector_env.yml
    ```

   This will create a Conda environment named `face_mask_env` (or whatever name is specified in `face_mask_detector_env.yml`) and install all the necessary dependencies.

4. **Activate the Conda environment:**

    ```bash
    $ conda activate face_mask_env
    ```

5. **Start using the project!**

    You're now ready to use the project within the activated Conda environment.

6. **Deactivate the Conda environment (after usage):**

    ```bash
    $ conda deactivate
    ```

   This step will deactivate the current Conda environment.

## Usage

### Demo

To run the demonstration:

1. **Activate the Conda environment:**

    ```bash
    $ conda activate face_mask_env
    ```

2. **Navigate to the demo directory:**

    ```bash
    $ cd demo
    ```

3. **Run the Python script:**

    ```bash
    $ python detect_mask.py
    ```

4. **Deactivate the Conda environment (after usage):**

    ```bash
    $ conda deactivate
    ```

   This step will deactivate the current Conda environment once you've completed using the script.

### Model

To train the model:

1. **Navigate to the demo directory:**

    ```bash
    $ cd model
    ```

2. **Download the dataset**

    [Download the dataset from Kaggle](https://www.kaggle.com/datasets/firebee/face-mask-detection-dataset/)

    Extract the dataset to the model directory

3. **Activate the Conda environment:**

    ```bash
    $ conda activate face_mask_env
    ```

3. **Run the Python script:**

    Make changes to the train.py file if you would like to edit the model
    ```bash
    $ python train.py
    ```

4. **Deactivate the Conda environment (after usage):**

    ```bash
    $ conda deactivate
    ```

   This step will deactivate the current Conda environment once you've completed using the script.

# License
This project is licensed under the MIT License.
