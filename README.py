# Solution-1
# Importing the libraries
import numpy as np
import pandas as pd
import keras


# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:,[2,4,5,6,7,9] ].values
y_train= dataset.iloc[:, 1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train[:, 2:])
X_train[:, 2:] = imputer.transform(X_train[:,2:])
print(X_train)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,[2,5]] = sc.fit_transform(X_train[:,[2,5]])


# Encoding categorical data
#Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()
X_train[:,1] = labelencoder_X1.fit_transform(X_train[:,1])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train), dtype=np.float)
X_train = X_train[:,0:]
print(X_train)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# opimizer details
opt = keras.optimizers.Adam(learning_rate=0.011)

# Compiling the ANN
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 20, epochs = 350)



# Part 3 - Making predictions and evaluating the model
#%% Importing Test Set

# Importing the dataset
dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:,[1,3,4,5,6,8] ].values

traveller_id = dataset2.iloc[:,0 ].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_test[:, 2:])
X_test[:, 2:] = imputer.transform(X_test[:,2:])
print(X_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test[:,[2,5]] = sc.fit_transform(X_test[:,[2,5]])

# Encoding categorical data
labelencoder_X2 = LabelEncoder()
X_test[:,1] = labelencoder_X2.fit_transform(X_test[:,1])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test), dtype=np.float)
X_test = X_test[:,0:]
print(X_test)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred2 = y_pred > .5
survivor_pred = (np.concatenate((traveller_id.reshape(len(traveller_id),1), y_pred2.reshape(len(y_pred2),1)),1))
print(survivor_pred)

#%%
survivor_pred2 = pd.DataFrame(data=survivor_pred, columns=('PassengerId','Survived')) 
survivor_pred2.to_csv('final2.csv',index=False)
