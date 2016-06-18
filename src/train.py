import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv('../dataset/train.csv')

print ds[:4]


# print ds.to_string()
#
ds['Sex'].replace('female', 1, inplace=True)
ds['Sex'].replace('male', 2, inplace=True)

print ds['Survived'].value_counts(dropna=False)
print ds['Fare'].value_counts(dropna=False)
# print ds['Parch'].value_counts(dropna=False)
ds['Age'].fillna(ds['Age'].mean(), inplace=True)



# ds['Embarked'].dropna(inplace=True)
# ds['Survived'].dropna(inplace=True)
# print 'prima'
# print ds['Age'].size
ds.dropna(subset=['Fare', 'Embarked'], inplace=True, how='any')
# print 'dopo'
# print ds['Age'].size

ds['Embarked'].replace('S', 1, inplace=True)
ds['Embarked'].replace('C', 2, inplace=True)
ds['Embarked'].replace('Q', 3, inplace=True)

def normalize(column):
    return (column - column.mean()) / (column.max() - column.min())


age_norm = normalize(ds['Age'])
pclass_norm = normalize(ds['Pclass'])
sex_norm = normalize(ds['Sex'])
sibsp_norm = normalize(ds['SibSp'])
parch_norm = normalize(ds['Parch'])
embarked_norm = normalize(ds['Embarked'])
fare_norm = normalize(ds['Fare'])

X_train = np.asarray(pd.concat([age_norm, pclass_norm, sex_norm, sibsp_norm, parch_norm, embarked_norm, fare_norm], axis=1, join='inner'))
Y_train = np.asarray(ds['Survived'])

print X_train
print Y_train

model = Sequential()

model.add(Dense(output_dim=10, input_dim=7))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.add(Activation("linear"))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

filepath = '../checkpoints/weights.best.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train[:600], Y_train[:600], callbacks=callbacks_list, nb_epoch=40, batch_size=32, verbose=0)

loss_history = history.history['loss']
acc_history = history.history['acc']
epochs = [(x + 1) for x in range(40)]

print epochs

ax = plt.subplot(211)
ax.plot(epochs, loss_history, color='red')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error Rate\n')
ax.set_title('Error Rate for Epoch\n')

ax2 = plt.subplot(212)
ax2.plot(epochs, acc_history, color='c')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy\n')
ax2.set_title('Accuracy for Epoch\n')

plt.subplots_adjust(hspace=0.8)

plt.show()

X_test = X_train[600:]
Y_test = Y_train[600:]

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print loss_and_metrics