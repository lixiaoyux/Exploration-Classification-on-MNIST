from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from dataPrep import dataMNIST
import numpy as np
import os


def ModelSelectK():
    # load the mnist dataset
    mnist = dataMNIST()

    # ranges of K
    ranges = 45 #110
    weights = [
        'uniform',
        'distance'
    ]
    metrics = [
        'euclidean',
        'manhattan',
        'chebyshev',
        'minkowski'
    ]

    # split dataset to train set and test set
    # using 25% for test
    train_data, test_data, train_target, test_target = train_test_split(np.array(mnist.data),
                                                                        mnist.target, test_size=0.25, random_state=1)

    # split train set to validation set
    # using 10% of train set for val
    train_data, val_data, train_target, val_target = train_test_split(np.array(train_data),
                                                                        train_target, test_size=0.10, random_state=2)

    print('train data size: {},\ntest data size: {},\neval data size: {}'.format(
        len(train_data), len(test_data), len(val_data)
    ))

    save_path = '../res'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    p_range = [
        3,
        4,
        5,
        6,
        7,
        #8
        #16,
        #32,
        #64
    ]

    labels = []
    kk = []
    test_scores = []
    for w in weights:
        # exclude minkowski
        for met in metrics[:len(metrics)-1]:
            labels.append(w[0] + "_" + met)
            eval_scores = []
            k_ranges = []
            for k in range(5, ranges):
                k_ranges.append(k)
                knn = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=w,
                    algorithm='auto',
                    metric=met
                )
                # train model
                knn.fit(train_data, train_target)

                # evaluate model
                score_eval = knn.score(val_data, val_target)
                print("k: {}, eval accuracy: {:.4f}".format(k, score_eval))
                eval_scores.append(score_eval)

            # get the best index of k
            idx = np.argmax(eval_scores)
            kk.append(k_ranges[idx])

            # test model
            model = KNeighborsClassifier(
                n_neighbors=k_ranges[idx],
                weights=w,
                algorithm='auto',
                metric=met
            )
            model.fit(train_data, train_target)

            score_test = model.score(test_data, test_target)
            print("k: {}, test accuracy: {:.4f}".format(k_ranges[idx], score_test))
            with open(os.path.join(save_path, 'res_KNN_2.txt'), 'a') as f:
                f.write('weight: ' + w + ', metric: ' + met +
                        "\nbest_k: {}, test accuracy: {:.4f}\n\n".format(k_ranges[idx], score_test))
            test_scores.append(score_test)

        # minkowski
        met = metrics[len(metrics) - 1]
        test_scores_mink = []
        kk_mink = []
        for m in p_range:
            eval_scores = []
            k_ranges = []
            for k in range(5, ranges):
                k_ranges.append(k)
                knn = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=w,
                    algorithm='auto',
                    metric=met,
                    p=m
                )
                # train model
                knn.fit(train_data, train_target)

                # evaluate model
                score_eval = knn.score(val_data, val_target)
                print("k: {}, eval accuracy: {:.4f}".format(k, score_eval))
                eval_scores.append(score_eval)

            # get the best index of k
            idx = np.argmax(eval_scores)
            kk_mink.append(k_ranges[idx])

            # test model
            model = KNeighborsClassifier(
                n_neighbors=k_ranges[idx],
                weights=w,
                algorithm='auto',
                metric=met,
                p=m
            )
            model.fit(train_data, train_target)

            score_test = model.score(test_data, test_target)
            print("k: {}, test accuracy: {:.4f}".format(k_ranges[idx], score_test))
            with open(os.path.join(save_path, 'res_KNN_2.txt'), 'a') as f:
                f.write('weight: ' + w + ', metric: ' + met + 'p: ' + str(m) +
                        "\nbest_k: {}, test accuracy: {:.4f}\n\n".format(k_ranges[idx], score_test))
            test_scores_mink.append(score_test)

        # select the maximum minkowski acc of p
        idx_p = np.argmax(test_scores_mink)
        best_p = p_range[idx_p]
        labels.append(w[0] + '_minkowski_' + str(best_p))
        test_scores.append(test_scores_mink[idx_p])
        kk.append(kk_mink[idx_p])

    # plot and save figures
    for x in range(len(labels)):
        plt.scatter(kk[x], test_scores[x], label=labels[x])       # len(test_scores) equals 4
    plt.xlabel('different metrics')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig('../res/accuracy_2.jpg')
    plt.show()
    plt.close()


if __name__ == '__main__':
    ModelSelectK()
