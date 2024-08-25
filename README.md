# Automated Essay Scoring with LSTM

This repository contains the code for an Automated Essay Scoring (AES) system using LSTM (Long Short-Term Memory) networks. The project is designed to grade essays based on multiple criteria such as content, coherence, grammar, and style, leveraging natural language processing (NLP) techniques and deep learning.

## Features

- Preprocessing text data with tokenization and padding
- LSTM-based deep learning model for essay scoring
- Save and load the tokenizer and padded sequences for easy reuse
- Model evaluation with accuracy and loss metrics

## Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Keras
- Numpy
- Pickle

### Installation

Clone this repository:

```bash
git clone https://github.com/alihassanml/Automated-Essay-Scoring-LSTM-Project.git
cd Automated-Essay-Scoring-LSTM-Project
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Usage

1. **Training the Model:**
   Run the training script to preprocess the data, build the LSTM model, and train it on the dataset.

2. **Saving the Tokenizer and Padded Sequences:**
   The project includes code to save and load the tokenizer and padded sequences using `pickle`. This allows for consistent preprocessing during inference.

3. **Evaluation:**
   Evaluate the model on the test set and get accuracy and loss metrics using:
   ```python
   loss, accuracy = model.evaluate(X_test, y_test)
   ```

### Notebook

For a step-by-step guide, you can explore the project in this [Kaggle Notebook](https://www.kaggle.com/code/alihassanml/automated-essay-scoring-lstm-project/notebook).

## Contributing

Feel free to submit issues or pull requests to improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **GitHub Repository:** [Automated-Essay-Scoring-LSTM-Project](https://github.com/alihassanml/Automated-Essay-Scoring-LSTM-Project.git)
- **Kaggle Notebook:** [Automated Essay Scoring LSTM Project](https://www.kaggle.com/code/alihassanml/automated-essay-scoring-lstm-project/notebook)



This README provides an overview of the project, instructions on getting started, and links to your GitHub repository and Kaggle notebook.
