from datasets import load_dataset

def download_dataset():
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    text = " ".join(dataset["train"]["text"])

    with open("data/dataset.txt", "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    download_dataset()