from Dataset.datasetReader import read_dataset
from keras.models import Model
from keras.layers import Conv2D, MaxPooling1D, Flatten, Dense, Embedding,\
                            Conv1D, concatenate, Input, BatchNormalization, Dropout, LSTM
from matplotlib import pyplot as plt


top_words = 5000
embedding_vector_length = 16
max_question_length = 30

train1, train2, train_label, test1, test2, test_label = read_dataset("Dataset/quora_duplicate_questions.tsv",
                                                                     train_sample_number=380000,
                                                                     max_question_length=max_question_length)
# train1, train2, train_label, test1, test2, test_label = read_dataset("Dataset/tsv.tsv",
#                                                                      train_sample_number=150,
#                                                                      max_question_length=max_question_length)

inputX1 = Input(shape=(max_question_length,))
inputX2 = Input(shape=(max_question_length,))

input_question = Input(shape=(max_question_length,))
x1 = Embedding(top_words, embedding_vector_length, input_length=max_question_length)(input_question)
x1 = Conv1D(filters=64, padding="same", kernel_size=3, use_bias=True, activation="sigmoid")(x1)
x1 = Dropout(0.4)(x1)
x1 = Conv1D(filters=32, padding="same", kernel_size=3, use_bias=True, activation="sigmoid")(x1)
x1 = Dropout(0.2)(x1)
x1 = MaxPooling1D()(x1)
x1 = LSTM(units=100, activation="sigmoid")(x1)
x1 = Dropout(0.2)(x1)
shared_network = Model(inputs=input_question, outputs=x1)

question1 = shared_network(inputX1)
question2 = shared_network(inputX2)

combined = concatenate([question1, question2])
z = Dense(units=100, activation="sigmoid")(combined)
z = Dense(units=1, activation="sigmoid")(z)
model = Model(inputs=[inputX1, inputX2], outputs=z)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
print(model.summary())

history = model.fit([train1, train2], train_label, validation_data=([test1, test2], test_label),
                    batch_size=128, epochs=20, verbose=2)
print(history.history.keys())

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

