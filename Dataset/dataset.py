# from keras.preprocessing.text import Tokenizer
# import csv
#
#
# with open("tsv.tsv") as tsvfile:
#     questions1 = []
#     questions2 = []
#     labels = []
#     reader = csv.DictReader(tsvfile, delimiter='\t')
#     for line in reader:
#         print(line['question1'])
#         questions1.append(line['question1'])
#         questions2.append(line['question2'])
#         labels.append(line['is_duplicate'])
#
#     print(questions1)
#
#     print ("============")
#     tokenizer = Tokenizer(num_words=20000)
#     tokenizer.fit_on_texts(questions1 + questions2)
#     # word_index = tokenizer.word_index
#
#     q1sequence = tokenizer.texts_to_sequences(questions1)
#     q2sequence = tokenizer.texts_to_sequences(questions2)
#
#     # word_index = tokenizer.word_index
#
#     print(q1sequence)
