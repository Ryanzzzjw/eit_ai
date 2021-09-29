import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.io import loadmat
import autokeras as ak
from sklearn import model_selection, metrics, preprocessing
import tensorflow as tf

def get_dataset(batch_size = 32, num_val_samples = 5000, test_size= 0.20, val_size=0.20):

    if test_size+val_size>=0.8:
        test_size= 0.2
        val_size=0.2


    # Load data
    Xih = loadmat(r"Xih_990x50k.mat") # <class 'numpy.ndarray'> 
    Xih = Xih["Xih"].T

    Yih = loadmat(r"Yih_990x50k.mat")
    Yih = Yih["Yih"].T

    print(np.shape(Xih) ,np.shape(Yih))

    X = tf.keras.utils.normalize(Xih, axis=0).astype("float32")
    # x_train = preprocessing.normalize(Xih)
    # y_train = preprocessing.normalize(Yih)
    Y = tf.keras.utils.normalize(Yih, axis=0).astype("float32")
    
    # make the 
    
    N_samples= len(X)

    SAMPLES_dataset= tf.data.Dataset.from_tensor_slices((X, Y))

    TRAIN_dataset= SAMPLES_dataset.take(int((1-test_size+val_size)*N_samples))
    TEST_dataset= SAMPLES_dataset.take(int(test_size*N_samples))
    VAL_dataset= SAMPLES_dataset.take(int(val_size*N_samples))

    # TRAIN_dataset= TRAIN_dataset.repeat().batch(batch_size)
    # TEST_dataset= TEST_dataset.repeat().batch(batch_size)
    # VAL_dataset=VAL_dataset.repeat().batch(batch_size)

    # Reserve num_val_samples samples for validation
  
    return (TRAIN_dataset, VAL_dataset, TEST_dataset)

# Xih = loadmat(r"Xih_990x50k.mat")
# Xih = Xih["Xih"].T
# print(Xih[0].size )

# Yih = loadmat(r"Yih_990x50k.mat")
# Yih = Yih["Yih"].T
# print(Yih[0].size)

#x_train, x_test, y_train, y_test = model_selection.train_test_split(Xih,Yih, test_size=0.2)
# strat = tf.distribute.MirroredStrategy()
# with strat.scope():
BATCH_SIZE=32
train_dataset, val_dataset, test_dataset = get_dataset()
print('here')
model = ak.StructuredDataRegressor(max_trials = 10, overwrite=True)
print('here')
# model.fit(train_dataset, epochs=200, validation_data=val_dataset, steps_per_epoch=(50000*0.6)//BATCH_SIZE, validation_steps= (50000*0.2)//BATCH_SIZE) #,epochs = 100
model.fit(train_dataset, epochs=200) #,epochs = 100

#_, acc = reg.evaluate(x_test,y_test)
#print('Accuracy: %.3f' % acc)

#y_predict = reg.predict(x_test)
#cm = metrics.confusio_matrix(y_test,y_predict)
#metrics.sns.heatmap(cm, annot = True) 

# model = model.export_model()
             
# print(model.summary())

# try:
#     model.save("model_autokeras_990el_allNORM", save_format="tf")
# except Exception:
#     model.save("model_autokeras_990el_allNORM.h5")

# predictions = model.predict(x_train)

# # Create triangulation.
# tr_data = loadmat(r"tr_data_990el.mat")
# xy = np.array(tr_data["nodes"])
# triangles = np.array(tr_data["tr_points"])

# x, y = xy.T

# print(x,y)
# print(triangles)

# triang = mtri.Triangulation(x, y)

# # Interpolate to regularly-spaced quad grid.
# conduct_elem = predictions[0]

# m = np.array(conduct_elem).flatten()
# n = np.array(triangles, dtype = float).flatten()

# print(m,n)

# z = [0 for a in range(np.size(x))]
# print(np.size(z))

# list = []

# for i in range (len(n)):
#     idx = n[int(i)] 
#     idx =int(idx)
#     #print(idx)
#     if idx not in list: 
#         j= int(i/3)
#         z[idx-1]=m[j]
#         list.append(idx)


# # Plot the triangulation.
# plt.plot()
# plt.tricontourf(triang, z)
# plt.triplot(triang, '-', alpha=.5)
# plt.title('Triangular grid')

# plt.tight_layout()
# plt.colorbar()

# plt.show()