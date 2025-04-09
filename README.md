# -Sentiment-Analysis-with-VADER-RoBERTa
# This notebook performs sentiment analysis on Amazon product reviews using two approaches:
- **VADER**: A rule-based sentiment analysis tool from NLTK
- **RoBERTa**: A transformer-based deep learning model via HuggingFace

##  Features

- Preprocess and analyze review sentiments from `Reviews.csv`
- Visualize sentiment distribution by star rating
- Compare outputs from VADER and RoBERTa
- Showcases model agreement/disagreement with plots and examples

##  Tools & Libraries Used

- `pandas`, `numpy` – Data processing
- `matplotlib`, `seaborn` – Data visualization
- `nltk` – VADER sentiment analysis
- `transformers` – Access RoBERTa model
- `tensorflow` and `scipy` – Model execution & softmax
- `tqdm` – For progress visualization


The notebook expects a file named `Reviews.csv` to be present in the same directory. This file should contain at least the following columns:
- `Id`
- `Score`
- `Text`

##  Usage

1. Install required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk tqdm transformers torch tensorflow
    ```

2. Download the VADER lexicon:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

3. Run all cells in order. The notebook will:
    - Load and visualize review scores
    - Run sentiment analysis using both VADER and RoBERTa
    - Compare their outputs
    - Visualize patterns and top predictions
