from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_data(filename):
    """
    Load dataset.
    
    :param filename: Filename of the data's `.npy` file.
    :returns: Dictionary of data points according to 
              https://github.com/hate-alert/HateXplain/blob/master/Data/README.md
    """

    encoder = LabelEncoder()
    encoder.classes_ = np.load(filename, allow_pickle=True)
    print(encoder.classes_)
    print(encoder)

def main():
    load_data("data/classes_two.npy")

if __name__ == "__main__":
    main()
