from text_preprocessing import preprocess_text, STOP_WORDS
from pandas import DataFrame, read_csv

if __name__ == "__main__":

    dataset = read_csv("dataset/IMDBDataset.csv")

    # Enoding labels
    dataset["sentiment"] = (
        dataset["sentiment"].replace("positive", 1).replace("negative", -1)
    )

    dataset["review"] = list(
        map(lambda x: preprocess_text(x, STOP_WORDS), dataset["review"])
    )

    print(dataset.head())

    dataset.to_csv("dataset/transformed_dataset.csv", index=False)
