import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.exceptions import MissingOptionalLibraryError
from sklearn.datasets import load_iris
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def iris(training_size, test_size, n, plot_data=False):#
    """ returns iris dataset """
    class_labels = [r'A', r'B', r'C']
    data, target = datasets.load_iris(return_X_y=True)
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, test_size=1, random_state=42)

    # Now we standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Scale to the range (-1,+1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    # Pick training size number of samples from each distro
    training_input = {key: (sample_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {key: (sample_test[label_test == k, :])[:test_size]
                  for k, key in enumerate(class_labels)}


    if not plot_data:
        try:
            import matplotlib.pyplot as plt
            
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname='Matplotlib',
                name='iris',
                pip_install='pip install matplotlib') from ex
        
        for k in range(0, 3):
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size])
        for k in range(0, 3):
            plt.scatter(sample_test[label_test== k, 0][:test_size],
                        sample_test[label_test == k, 1][:test_size],c="blue")
        
        '''
        ax = plt.gca()
        ax.set_aspect(1)
          
        colours = ["#ec6f86", "#4573e7", "#ad61ed"]

        c_transparent = "#00000000"
        custom_lines = [ #右邊的表格
        Patch(facecolor=colours[0], edgecolor=c_transparent, label="Class 0"),
        Patch(facecolor=colours[1], edgecolor=c_transparent, label="Class 1"),
        Patch(facecolor=colours[2], edgecolor=c_transparent, label="Class 2"),
        Line2D([0], [0], marker="o", color=c_transparent, label="Train",
               markerfacecolor="black", markersize=10),
        Line2D([0], [0], marker="x", color=c_transparent, label="Test",
               markerfacecolor="black", markersize=10),
        ]
        ax.legend(handles=custom_lines, bbox_to_anchor=(1.0, 0.75))
'''
        plt.title("Iris dataset")
        plt.show()
        
    '''print(f"""sample_train: {sample_train}
        training_input: {training_input}
        test_input: {test_input}
        class_labels: {class_labels}
        """)'''
    
    return sample_train, training_input, test_input, class_labels

iris(150,2,4)
