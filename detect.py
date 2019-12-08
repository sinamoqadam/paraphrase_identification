from Dataset.datasetReader import read_dataset
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Embedding, Conv1D, concatenate, Input


# train1, train2, train_label, test1, test2, test_label = read_dataset("Dataset/quora_duplicate_questions.tsv",
#                                                                      train_sample_number=350000)


model = Sequential()

top_words = 20000
embedding_vector_length = 32
max_question_length = 100

train1, train2, train_label, test1, test2, test_label = read_dataset("Dataset/tsv.tsv",
                                                                     train_sample_number=150,
                                                                     max_question_length=max_question_length)

inputX1 = Input(shape=(max_question_length,))
inputX2 = Input(shape=(max_question_length,))

x1 = Embedding(top_words, embedding_vector_length, input_length=max_question_length)(inputX1)
x1 = Conv1D(filters=32, kernel_size=3, use_bias=True, activation="relu")(x1)
x1 = Model(inputs=inputX1, outputs=x1)

x2 = Embedding(top_words, embedding_vector_length, input_length=max_question_length)(inputX2)
x2 = Conv1D(filters=32, kernel_size=3, use_bias=True, activation="relu")(x2)
x2 = Model(inputs=inputX2, outputs=x2)

combined = concatenate([x1.output, x2.output])
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="sigmoid")(z)

model = Model(inputs=[x1.input, x2.input], outputs=z)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
model.fit([train1, train2], train_label, validation_data=([test1, test2], test_label),
          batch_size=16, epochs=6, verbose=1)

