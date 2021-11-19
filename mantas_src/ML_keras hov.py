import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.io import loadmat
import tensorflow as tf
import tensorflow.keras as keras
import horovod.tensorflow.keras as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')




strategy = tf.distribute.get_strategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

Xih = loadmat(r"Xih_990x50k.mat")
Xih = Xih["Xih"].T
print(Xih[0].size )

Yih = loadmat(r"Yih_990x50k.mat")
Yih = Yih["Yih"].T
print(Yih[0].size)

x_train = tf.keras.utils.normalize(Xih, axis=1)

#x_test = tf.keras.utils.normalize(x_test, axis=1)
y_train = Yih

#x_train = x_train.map(x_train)
input_len= Xih[0].size
output_len= Yih[0].size


model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=(Xih[1].shape,)))
model.add(tf.keras.layers.Dense(input_len, input_dim = input_len, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(output_len, activation=tf.nn.sigmoid)) 

model.output_shape

opt =tf.optimizers.Adam(0.001 * hvd.size())
# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                experimental_run_tf_function=False)

model.summary()
print(x_train.shape , y_train.shape )

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

model.fit(x_train, y_train, epochs=700 // hvd.size(), validation_split=0.2, callbacks=callbacks, verbose= 1 if hvd.rank() == 0 else 0)
#x_train = x_train.reshape(-1, 256)

model.save('test_keras.model')

predictions = model.predict(x_train)
Yih[0] = predictions[0]


# Create triangulation.
tr_data = loadmat(r"tr_data_990el.mat")
xy = np.array(tr_data["nodes"])
triangles = np.array(tr_data["tr_points"])

print(xy)

x, y = xy.T

print(x,y)

print(triangles)
triang = mtri.Triangulation(x, y)

# Interpolate to regularly-spaced quad grid.
conduct_elem = Yih[0]


m = np.array(conduct_elem).flatten()
n = np.array(triangles, dtype = float).flatten()

print(m,n)

z = [0 for a in range(np.size(x))]
print(np.size(z))

list = []

for i in range (len(n)):
    idx = n[int(i)] 
    idx =int(idx)
    #print(idx)
    if idx not in list: 
        j= int(i/3)
        z[idx-1]=m[j]
        list.append(idx)


# Plot the triangulation.
plt.plot()
plt.tricontourf(triang, z)
plt.triplot(triang, '-', alpha=.5)
plt.title('Triangular grid')

plt.tight_layout()
plt.colorbar()

plt.show()