from sklearn import datasets

"""
    load the MNIST datasets using sklearn lib
    other datasets included:
        datasets.load_iris
        datasets.fetch_lfw_people
        ...
"""
def dataMNIST():

    data = datasets.load_digits()
    return data
