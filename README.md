# Crop Recommendation System 🌾

Welcome to the Crop Recommendation System! This application helps farmers and agriculturists determine the most suitable crop to grow based on various soil and weather parameters.

## Features

- **User Input**: Adjust parameters such as Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, and rainfall to get a crop recommendation.
- **Crop Recommendation**: Displays the recommended crop based on the input parameters.
- **Prediction Probabilities**: Shows a bar chart of the probabilities for each crop, indicating the likelihood of being the most suitable crop to grow.

## Demo

You can access the live application [here](https://crop-recommendation-system-application.streamlit.app/).

## Installation

To run this application locally, you need to have Python installed on your machine. Follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/crop-recommendation-system.git
    cd crop-recommendation-system
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Files

- `app.py`: The main application file.
- `Crop_recommendation.csv`: The dataset used for training the model.
- `requirements.txt`: List of Python packages required to run the app.

## Usage

1. Open the app in your browser by running the command:
    ```bash
    streamlit run app.py
    ```

2. Use the sliders in the sidebar to input the soil and weather parameters.

3. View the recommended crop and the prediction probabilities in the main panel.

## Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`

Make sure these dependencies are listed in your `requirements.txt` file.

---

&copy; 2024 Cuckoo Labs. All rights reserved.
