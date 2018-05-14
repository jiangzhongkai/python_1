import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
# import random


train_losses = np.loadtxt('./train_losses2.0.txt')
test_losses = np.loadtxt('./test_losses2.0.txt')

train_losses2 = np.loadtxt('./train_losses2.1.txt')
test_losses2 = np.loadtxt('./test_losses2.1.txt')

epoch=1000





font = {
	'family' : 'Bitstream Vera Sans',
	'weight' : 'bold',
	'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

# indep_train_axis = np.array(range(config.batch_size, (len(train_losses)+1)*config.batch_size, config.batch_size))
plt.plot(range(1,epoch+1), np.array(train_losses),     "b--", label="Train losses")
plt.plot(range(1,epoch+1), np.array(train_losses2), "g--", label="Train losses with Dropout")

# indep_test_axis = np.array(range(config.batch_size, len(test_losses)*config.display_iter, config.display_iter)[:-1] + [config.train_count*config.training_epochs])
plt.plot(range(1,epoch+1), np.array(test_losses),     "b-", label="Test losses")
plt.plot(range(1,epoch+1), np.array(test_losses2), "g-", label="Test losses with Dropout")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress Loss')
plt.xlabel('Training iteration')

plt.show()