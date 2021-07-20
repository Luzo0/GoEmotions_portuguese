from io import BytesIO
import os
import requests
import pandas as pd
from tqdm import tqdm
import time
from itranslate import itranslate as itrans

MAX_LENGTH = 5000  # google translate api max text length is 5000 characters per request
WAIT_TIME = 5  # wait time in seconds between each requisition to google translate api
OUTPUT_DIR = 'dataset'

if __name__ == "__main__":

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    for string in ['train', 'dev', 'test']:
        url = f'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/{string}.tsv'
        r = requests.get(url)
        r.raise_for_status()

        df = pd.read_csv(BytesIO(r.content), sep="\t", names=['text', 'labels', 'id'])

        sentences = []
        for sentence in df['text']:
            sentences.append(sentence)

        chunks = []
        chunk = sentences[0]

        for sentence in sentences[1:]:
            if len((chunk + '\n\r\n' + sentence).encode('utf-8')) < MAX_LENGTH:
                chunk += '\n\r\n' + sentence
            else:
                chunks.append(chunk)
                chunk = sentence
            if sentence == sentences[-1]:
                chunks.append(chunk)

        translated_chunks = []

        print('Translating {}.tsv'.format(string))
        for chunk in tqdm(chunks):
            translated_chunks.append(itrans(chunk, to_lang="pt"))
            time.sleep(WAIT_TIME)

        translated = '\n\r\n'.join(translated_chunks)
        translated = translated.split('\n\r\n')

        df['text'] = translated

        output_file = os.path.join(OUTPUT_DIR, '{}.tsv'.format(string))
        df.to_csv(output_file, sep='\t')

    print('All files were downloaded and translated successfully')
