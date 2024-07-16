import os
import stanza
from stanza.models.classifiers.data import SentimentDatum
from stanza.utils.datasets.sentiment.process_utils import write_list

# Download the Polish model if not already downloaded
stanza.download('pl')

# Define label conversion
def convert_label(label):
    label_map = {
        "__label__z_minus_m": 0,
        "__label__z_zero": 1,
        "__label__z_amb": 2,
        "__label__z_plus_m": 3
    }
    return label_map.get(label, -1)  # Return -1 if the label is not found

# Read sentences and labels from the file
def read_sentences_and_labels(filename):
    sentiment_data = []
    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            pieces = line.rsplit(" ", 1)
            if len(pieces) < 2:
                continue
            text, label = pieces
            label = convert_label(label.strip())
            if label == -1:
                continue
            sentiment_data.append(SentimentDatum(label, text.strip()))
    return sentiment_data

# Tokenize the sentences
def tokenize(sentiment_data, pipe):
    docs = [datum.text for datum in sentiment_data]
    out_docs = pipe(docs)

    if isinstance(out_docs, list):
        tokenized_data = [
            SentimentDatum(datum.sentiment, [token.text for token in doc.sentences[0].tokens])
            for datum, doc in zip(sentiment_data, out_docs)
        ]
    else:  # Handle the case where out_docs is a single Document object
        tokenized_data = [
            SentimentDatum(datum.sentiment, [token.text for token in out_docs.sentences[i].tokens])
            for i, datum in enumerate(sentiment_data)
        ]

    return tokenized_data

# Process the dataset
def process_dataset(file_paths, pipe):
    train_data = read_sentences_and_labels(file_paths['train'])
    dev_data = read_sentences_and_labels(file_paths['dev'])
    test_data = read_sentences_and_labels(file_paths['test'])

    train_data = tokenize(train_data, pipe)
    dev_data = tokenize(dev_data, pipe)
    test_data = tokenize(test_data, pipe)

    return train_data, dev_data, test_data

def main():
    # Define file paths
    base_path = '/u/nlp/data/sentiment/stanza/polish/PolEmo2.0/dataset_conll/dataset_conll'
    file_paths = {
        'train': os.path.join(base_path, 'all.sentence.train.txt'),
        'dev': os.path.join(base_path, 'all.sentence.dev.txt'),
        'test': os.path.join(base_path, 'all.sentence.test.txt')
    }

    # Initialize the pipeline
    pipe = stanza.Pipeline(lang="pl", processors="tokenize", tokenize_no_ssplit=True)

    # Process the dataset
    train_data, dev_data, test_data = process_dataset(file_paths, pipe)

    # Get output directory from environment variable
    out_directory = os.getenv('SENTIMENT_DATA_DIR', 'processed_data')
    os.makedirs(out_directory, exist_ok=True)

    # Write the dataset to JSON files
    write_list(os.path.join(out_directory, 'pl_polemo2.train.json'), train_data)
    write_list(os.path.join(out_directory, 'pl_polemo2.dev.json'), dev_data)
    write_list(os.path.join(out_directory, 'pl_polemo2.test.json'), test_data)

    print(f"Data processing complete. Processed files are saved in {out_directory}")

if __name__ == '__main__':
    main()
