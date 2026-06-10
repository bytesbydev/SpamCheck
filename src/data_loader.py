import pandas as pd


def load_data(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df["label"] = df["label"].str.lower()
    df["message"] = df["message"].str.strip()

    return df.reset_index(drop=True)