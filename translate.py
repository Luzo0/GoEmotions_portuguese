from io import BytesIO
import os
import requests
import pandas as pd
from tqdm import tqdm
import time
from googletrans import Translator

MAX_LENGTH = 5000  # Max text length per request
WAIT_TIME = 0      # Wait time between requests (in seconds)
OUTPUT_DIR = 'dataset'
MAX_RETRIES = 3    # Maximum number of retries for a failed translation

if __name__ == "__main__":

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    translator = Translator()

    for string in ['train', 'dev', 'test']:
        url = f'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/{string}.tsv'
        r = requests.get(url)
        r.raise_for_status()

        df = pd.read_csv(BytesIO(r.content), sep="\t", names=['text', 'labels', 'id'])

        sentences = df['text'].tolist()

        chunks = []
        chunk = [sentences[0]]

        for sentence in sentences[1:]:
            estimated_size = len(' '.join(chunk + [sentence]).encode('utf-8'))
            if estimated_size < MAX_LENGTH:
                chunk.append(sentence)
            else:
                chunks.append(chunk)
                chunk = [sentence]
        chunks.append(chunk)

        translated_sentences = []

        print('Translating {}.tsv'.format(string))
        for chunk in tqdm(chunks):
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    translated = translator.translate(chunk, dest='pt')
                    translated_texts = [t.text for t in translated]
                    translated_sentences.extend(translated_texts)
                    break
                except Exception as e:
                    retries += 1
                    print(f"Error during translation: {e}. Retrying ({retries}/{MAX_RETRIES})...")
                    time.sleep(WAIT_TIME)
            else:
                print("Max retries exceeded. Using original sentences for this chunk.")
                translated_sentences.extend(chunk)
            time.sleep(WAIT_TIME)

        df['text'] = translated_sentences

        output_file = os.path.join(OUTPUT_DIR, '{}.tsv'.format(string))
        df.to_csv(output_file, sep='\t', index=False)

    print('All files were downloaded and translated successfully')
