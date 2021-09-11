import matplotlib.pyplot as plt
import numpy as np
from qiskit.ml.datasets import breast_cancer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.algorithms import QSVM
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.algorithms import SklearnSVM
import pandas as pd
import random as rand

seed = 10599
aqua_globals.random_seed = seed
feature_dim = df.shape[1]-1 #控制幾維的data，目前都是抓最大的維度


'''
		        type	 data				         size
sample_total    ndarray  [-0.8333448  -0.25876499]	 769   (全資料)
training_input  dict	 從sample_tota A挖10 B挖10筆  20
test_input	    dict	 從sample_tota A挖5 B挖5筆 	 10	
class_labels	list	 label ['A','B'] 良性 惡性	  2
'''

#把CSV內的資料挖出來
df = pd.read_csv("./breast_cancer_source.csv", header=0)

for count in range(0,1): #可以跑多次看random的情況
    array_temp= []
    dict_malignant = []
    dict_benign = []
    columns = feature_dim #df.shape[rows, columns] df.shape[1]-1
    array1_temp= []
    for i in range(0,len(df)):
        #print(df.iloc[i])
        if df.iloc[i][df.shape[1]-1] == 0:  #columns
            array_temp.append(df.iloc[i])
            for j in range(0,columns):
              array1_temp.append(array_temp[0][j])
            dict_malignant.append(array1_temp)
        else:
            array_temp.append(df.iloc[i])
            for j in range(0,columns):
              array1_temp.append(array_temp[0][j])
            dict_benign.append(array1_temp)
        array_temp = []
        array1_temp = []

 #製作training/test_input的資料   
    training_input = {}
    test_input={}
    array_temp = rand.sample(dict_malignant,15)
    array1_temp = rand.sample(dict_benign,15)
    array2_temp =[]
    array3_temp =[]
    #training_input setdata
    for i in range(0,9):
        array2_temp.append(array_temp[i])
        array3_temp.append(array1_temp[i])
    training_input['A'] = np.array(array2_temp)
    training_input['B'] = np.array(array3_temp)

    array2_temp =[]
    array3_temp =[]

    #test_input setdata
    for i in range(10,14):
        array2_temp.append(array_temp[i])
        array3_temp.append(array1_temp[i])
    test_input['A'] = np.array(array2_temp)
    test_input['B'] = np.array(array3_temp)
    class_labels = ['A', 'B']


    #QSVM方法
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=feature_dim, entanglement='linear')
    qsvm = QSVM(feature_map, training_input, test_input)

    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

    result = qsvm.run(quantum_instance)

    print(f'Testing success ratio: {result["testing_accuracy"]}')
    kernel_matrix = result['kernel_matrix_training']
    img = plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')
    #傳統方法 因為用user denfined dataset 所以用SKlearn差到爆是很正常
    #result = SklearnSVM(training_input, test_input).run()
    #print(f'Testing success ratio: {result["testing_accuracy"]}')
