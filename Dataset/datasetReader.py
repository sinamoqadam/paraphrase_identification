from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from random import shuffle
import csv


def read_tsv(dataset_path):
    dataset = []
    questions1 = []
    questions2 = []
    labels = []
    with open(dataset_path) as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for line in reader:
            dataset.append([line['question1'], line['question1'], line['is_duplicate']])

    shuffle(dataset)

    for i in range(len(dataset)):
        questions1.append(dataset[i][0])
        questions2.append(dataset[i][1])
        labels.append(int(dataset[i][2]))

    return questions1, questions2, labels


def read_dataset(dataset_path, train_sample_number, max_question_length=50):
    questions1, questions2, labels = read_tsv(dataset_path)
    assert len(questions1) == len(labels)
    assert len(questions2) == len(labels)

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(questions1 + questions2)

    q1sequence = tokenizer.texts_to_sequences(questions1)
    q2sequence = tokenizer.texts_to_sequences(questions2)

    q1sequence = sequence.pad_sequences(q1sequence, maxlen=max_question_length)
    q2sequence = sequence.pad_sequences(q2sequence, maxlen=max_question_length)

    train1 = q1sequence[:train_sample_number]
    train2 = q2sequence[:train_sample_number]
    train_label = labels[:train_sample_number]

    test1 = q1sequence[train_sample_number:]
    test2 = q2sequence[train_sample_number:]
    test_label = labels[train_sample_number:]

    return train1, train2, train_label, test1, test2, test_label
