import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

def bin_to_index(data):
    index_data = np.argmax(data, axis=1)
    return index_data

def results_report(results_history_table, final_results, actual, predictions, shape_labels):
    # Plot loss and accuracy on each iteration
    loss_accuracy = pd.DataFrame(results_history_table)
    loss_accuracy.plot()

    # Final loss and accuracy
    print("Final loss and accuracy")
    print(final_results)

    # Print first 10 predictions and actual, raw
    print("First 10 predictions")
    print(predictions.round(decimals=3)[:10])
    print("First 10 actual values")
    print(actual[:10])

    # Get them in index form
    actual_indices = bin_to_index(actual)
    prediction_indices = bin_to_index(predictions)   
    
    # Confusion matrix
    print("Confusion matrix")
    print(confusion_matrix(actual_indices, prediction_indices)) #add labels


def accuracy_compare(results):
    for model, accuracy in results:
        print(model + " accuracy: " + str(accuracy))

# All models are 4 layers sequential

# Basic model
def basic_model(data, out_layer_activation, num_epochs, labels):
    X_train, Y_train, X_test, Y_test = data
    model = Sequential()

    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(4,activation=out_layer_activation)) 

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    print("Basic model: ")
    print("Output layer activation function: " + out_layer_activation)
    print("Epochs: " + str(num_epochs))
    print("Running basic model... ")

    model.fit(x=X_train, y=Y_train, epochs=num_epochs, validation_data=(X_test, Y_test))

    loss_acc_over_iterations = pd.DataFrame(model.history.history)
    final_loss_acc = model.evaluate(X_test, Y_test)
    test_predictions = model.predict(X_test)

    results_report(loss_acc_over_iterations, final_loss_acc, Y_test, test_predictions, labels)

    model_name = "Basic model " + out_layer_activation
    accuracy = final_loss_acc[1]
    return (model_name, accuracy)

#early stop model
def early_stop_model(data, out_layer_activation, num_epochs, patience_val, labels):
    
    X_train, Y_train, X_test, Y_test = data

    model = Sequential()

    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(4,activation=out_layer_activation)) 

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    #early stop
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=patience_val)

    print("Early stop model: ")
    print("Output layer activation function: " + out_layer_activation)
    print("Epochs: " + str(num_epochs))
    print("Patience: " + str(patience_val))
    print("Running early stop model... ")

    model.fit(x=X_train, y=Y_train, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[early_stop])

    loss_acc_over_iterations = pd.DataFrame(model.history.history)
    final_loss_acc = model.evaluate(X_test, Y_test)
    test_predictions = model.predict(X_test)

    results_report(loss_acc_over_iterations, final_loss_acc, Y_test, test_predictions, labels)

    model_name = "Early stop model " + out_layer_activation
    accuracy = final_loss_acc[1]
    return (model_name, accuracy)

#model with early stop and dropouts on 2 hidden layers
#early stop model
def dropout_model(data, out_layer_activation, num_epochs, patience_val, labels):
    X_train, Y_train, X_test, Y_test = data

    model = Sequential()

    model.add(Dense(100,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation=out_layer_activation)) 

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    #early stop
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=patience_val)

    print("50/% /dropout model: ")
    print("Output layer activation function: " + out_layer_activation)
    print("Epochs: " + str(num_epochs))
    print("Patience: " + str(patience_val))
    print("Running early stop and dropouts model... ")

    model.fit(x=X_train, y=Y_train, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[early_stop])

    loss_acc_over_iterations = pd.DataFrame(model.history.history)
    final_loss_acc = model.evaluate(X_test, Y_test)
    test_predictions = model.predict(X_test)

    results_report(loss_acc_over_iterations, final_loss_acc, Y_test, test_predictions, labels)

    model_name = "Early stop and dropout model " + out_layer_activation
    accuracy = final_loss_acc[1]
    return (model_name, accuracy)

def main():
    shapes = ['sine', 'square','sawtooth','burst']
    df = pd.read_csv('datasets/tables/full_data.csv', header=None)  

    X = df.drop(df.columns[-4:],axis=1).values
    Y = df.iloc[:,-4:].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    a = basic_model((X_train, Y_train, X_test, Y_test), "sigmoid", 150, shapes)
    b = basic_model((X_train, Y_train, X_test, Y_test), "softmax", 150, shapes)
    c = early_stop_model((X_train, Y_train, X_test, Y_test), "sigmoid", 200, 50, shapes)
    d = dropout_model((X_train, Y_train, X_test, Y_test),  "sigmoid", 200, 50, shapes)

    results = (a,b,c,d)
    accuracy_compare(results)

main()