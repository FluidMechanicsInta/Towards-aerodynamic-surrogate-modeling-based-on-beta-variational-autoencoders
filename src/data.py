import numpy as np
import pickle
from sklearn.decomposition import PCA


def load_filtered_dataset(path):

    print("Getting dataset...")

    with open(path, "rb") as handle:
        filtered_dataset = pickle.load(handle)

    Cp_train = filtered_dataset["Cp_train_filtered"]
    Cp_test = filtered_dataset["Cp_test_filtered"]

    Cp_mean = filtered_dataset["Cp_mean_filtered"]
    Cp_std = filtered_dataset["Cp_std_filtered"]

    MA_train = filtered_dataset["MA_train_filtered"]
    MA_test = filtered_dataset["MA_test_filtered"]

    MA_mean = np.mean(MA_train)
    MA_std = np.std(MA_train)

    print("Done")

    return (MA_train, MA_test, Cp_train, Cp_test,
            MA_mean, MA_std, Cp_mean, Cp_std)

def compute_pca(cp_train, cp_test):

    print('Computing PCA')
    pca = PCA()
    pca.fit(cp_train)

    PCA_train = pca.transform(cp_train)
    PCA_test = pca.transform(cp_test)
    print('Computed PCA')

    PCA_mean = np.mean(PCA_train)
    PCA_std = np.std(PCA_train)

    return PCA_train, PCA_test, PCA_mean, PCA_std

if __name__ == "__main__":

    (MA_train, MA_test, Cp_train, Cp_test,
     MA_mean, MA_std, Cp_mean, Cp_std) = load_filtered_dataset(path="../00_data/filtered_dataset.pkl")
