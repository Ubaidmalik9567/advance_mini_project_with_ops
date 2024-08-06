import pandas as pd
import pathlib
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetching_data(url: str) -> pd.DataFrame:
    try:
        logging.info(f"Fetching data from {url}")
        dataset = pd.read_csv(url)
        logging.info("Data fetched successfully")
        return dataset
    except Exception as e:
        logging.error(f"Error fetching data from {url}: {e}")
        raise

def save_data(data: pd.DataFrame, path: str) -> None:
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        data.to_csv(path + "/raw.csv", index=False, header=True)
        logging.info(f"Data saved successfully at {path}/raw.csv")
    except Exception as e:
        logging.error(f"Error saving data to {path}/raw.csv: {e}")
        raise

def main():
    try:
        current_dir = pathlib.Path(__file__)
        home_dir = current_dir.parent.parent.parent

        path = sys.argv[1]
        data_save_path = home_dir.as_posix() + path
#"G:/dataset/tweets.csv" 
        link = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        df = fetching_data(link)
        save_data(df, data_save_path)
    except Exception as e:
        logging.critical(f"Critical error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

