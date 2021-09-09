import argparse
import pytorch_lightning as pl
import json
from transformers import AutoTokenizer
import torch
from utils import prepare_datasets, labels_decode
from sklearn.preprocessing import MultiLabelBinarizer
from model import EmotionDataModule, EmotionTagger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir", type=str, required=True,
                        help='The output directory to which the model is saved')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='Batch size for training (default = 16)')
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help='Maximum sequence length (default = 128)')
    parser.add_argument("--model_name", type=str, default="neuralmind/bert-base-portuguese-cased",
                        help='Name of the model to be used (default = "neuralmind/bert-base-portuguese-cased")')
    parser.add_argument("--n_epochs", type=int, default=4,
                        help='Number of epochs to run fine tuning (default = 4)')
    parser.add_argument("--warmup_proportion", type=float, default=0.2,
                        help='Float number between 0 and 1 that represents the proportion for warmup (default = 0.2)')
    parser.add_argument("--beta", type=float, default=0.999,
                        help='Beta parameter for weighing method Class-Balanced Loss (default = 0.999)')
    parser.add_argument("--no_cuda", type=bool, default=False,
                        help='If passed True, gpu will not be used (default = False)')
    parser.add_argument("--seed", type=int, default=42,
                        help='Seed for pseudo-random number generation for pytorch, numpy, python.random (default = 42)')
    parser.add_argument("--taxonomy", type=str, default='original',
                        help='Select which taxonomy to be used original or ekman (default = "original")')

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    with open('emotion_dict.json', 'r', encoding='utf-8') as emotion_dict:
        emotion_dict = json.loads(emotion_dict.read())
    with open('ekman_mapping.json', 'r', encoding='utf-8') as ekman_mapping:
        ekman_mapping = json.loads(ekman_mapping.read())

    if args.taxonomy == 'original':
        label_names = list(emotion_dict.keys())
    if args.taxonomy == 'ekman':
        label_names = list(ekman_mapping.keys())

    n_classes = len(label_names)

    df_train, df_validation, df_test = prepare_datasets('dataset', n_classes, args.taxonomy)

    sent_freq = {i: 0 for i in range(0, n_classes)}
    for label in df_train['labels']:
        mlb = MultiLabelBinarizer()
        mlb.fit([[x for x in range(0, n_classes)]])
        label = labels_decode(label, mlb)
        for i in sent_freq.keys():
            for j in label:
                if j == i:
                    sent_freq[i] += 1

    steps_per_epoch = len(df_train) // args.batch_size
    total_training_steps = steps_per_epoch * args.n_epochs
    warmup_steps = total_training_steps * args.warmup_proportion
    samples_per_classes = [x for x in sent_freq.values()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_module = EmotionDataModule(
        df_train,
        df_validation,
        df_test,
        tokenizer,
        batch_size=args.batch_size,
        max_token_len=args.max_seq_length
    )

    model = EmotionTagger(
        n_classes=n_classes,
        model_name=args.model_name,
        samples_per_classes=samples_per_classes,
        beta=args.beta,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        no_cuda=args.no_cuda
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="bert-checkpoints",
        filename=f"best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args.n_epochs,
        gpus=1 if torch.cuda.is_available() and not args.no_cuda else 0,
        progress_bar_refresh_rate=30
    )

    if args.taxonomy == 'original':
        model.bert.config.id2label = {str(id_): label for id_, label in enumerate(emotion_dict)}
        model.bert.config.label2id = {label: id_ for id_, label in enumerate(emotion_dict)}
    if args.taxonomy == 'ekman':
        model.bert.config.id2label = {str(id_): label for id_, label in enumerate(ekman_mapping)}
        model.bert.config.label2id = {label: id_ for id_, label in enumerate(ekman_mapping)}
        
    trainer.fit(model, data_module)

    model_to_save = model.bert
    model_to_save.save_pretrained(args.model_output_dir)
    tokenizer.save_pretrained(args.model_output_dir)
