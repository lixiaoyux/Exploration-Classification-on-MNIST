from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from dataPrep import dataMNIST
import numpy as np
import os


def SVM_model():
    # get mnist dataset
    mnist = dataMNIST()

    train_data, test_data, train_target, test_target = train_test_split(mnist.data,
                                                                        mnist.target, test_size=0.2, random_state=1)
    train_data, val_data, train_target, val_target = train_test_split(train_data,
                                                                      train_target, test_size=0.2, random_state=2)
    print('train data size: {},\ntest data size: {},\neval data size: {}'.format(
        len(train_data), len(test_data), len(val_data)
    ))
    save_path = '../res'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    c_value = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    scores_test = []
    scores_val = []
    for c in c_value:
        model = svm.SVC(C=c, kernel='rbf', gamma='scale',
                        decision_function_shape='ovr')
        model.fit(train_data, train_target)
        score_test = model.score(test_data, test_target)
        score_val = model.score(val_data, val_target)
        scores_test.append(score_test)
        scores_val.append(score_val)
        print("test accuracy: {:.5f}\n validation accuracy: {:.5f}\n".format(
            score_test, score_val
        ))
        with open(os.path.join(save_path, 'res_SVM_C.txt'), 'a') as f:
            f.write("C-"+ str(c) + "\ntest accuracy: {:.5f}\n validation accuracy: {:.5f}\n\n".format(
            score_test, score_val
        ))
    max_test = np.argmax(scores_test)
    max_val = np.argmax(scores_val)
    print("max acc of test res: " + str(c_value[max_test]))
    print("max acc of val res: " + str(c_value[max_val]))

    # plot
    plt.plot(c_value, scores_test, label='test_acc')
    plt.plot(c_value, scores_val, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('c-values')
    plt.ylabel('accuracy')
    plt.plot()
    plt.savefig('../res/acc_C-values.jpg')
    plt.show()


if __name__ == '__main__':
    SVM_model()