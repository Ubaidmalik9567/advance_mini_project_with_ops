import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import yaml
import sys
import pathlib
from sklearn.model_selection import train_test_split
import logging
import nltk

# nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocessing(raw_dataset_path: str, split_size: float, seed: float):
    try:
        df = pd.read_csv(raw_dataset_path)
        logging.info("Loaded raw dataset successfully.")
    except Exception as e:
        logging.error(f"Failed to load raw dataset: {e}")
        raise

    try:
        df = df.drop(columns=['tweet_id'])
        logging.info("Dropped 'tweet_id' column.")
    except KeyError as e:
        logging.error(f"Failed to drop 'tweet_id' column: {e}")
        raise

    try:
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df.dropna(inplace=True)
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logging.info("Filtered and processed sentiment data.")
    except Exception as e:
        logging.error(f"Error processing sentiment data: {e}")
        raise

    try:
        train_data, test_data = train_test_split(final_df, test_size=split_size, random_state=seed)
        logging.info("Split data into training and test sets.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df):
    try:
        df.content = df.content.apply(lower_case)
        df.content = df.content.apply(remove_stop_words)
        df.content = df.content.apply(removing_numbers)
        df.content = df.content.apply(removing_punctuations)
        df.content = df.content.apply(removing_urls)
        df.content = df.content.apply(lemmatization)
        logging.info("Normalized text data.")
        return df
    except Exception as e:
        logging.error(f"Error normalizing text: {e}")
        raise

def save_data(traindata: pd.DataFrame, testdata: pd.DataFrame, path: str) -> None:
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        traindata.to_csv(path + "/preprocess_traindata.csv", index=False, header=True)
        testdata.to_csv(path + "/preprocess_testdata.csv", index=False, header=True)
        logging.info("Saved processed data successfully.")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main():
    try:
        current_dir = pathlib.Path(__file__)
        home_dir = current_dir.parent.parent.parent

        path = sys.argv[1]
        predata_save_path = home_dir.as_posix() + path + "/interim"
        raw_dataset_path = home_dir.as_posix() + path + "/raw/raw.csv"
        
        params_location = home_dir.as_posix() + '/params.yaml'
        parameters = yaml.safe_load(open(params_location))["data_transformation"]

        train_data, test_data = preprocessing(raw_dataset_path, parameters["split_dataset_size"], parameters["seed"])
        pretrain_data = normalize_text(train_data)
        pretest_data = normalize_text(test_data)
        save_data(pretrain_data, pretest_data, predata_save_path)
        logging.info("Data processing pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()

