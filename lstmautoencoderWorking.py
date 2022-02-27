

import tensorflow as tf


from os import listdir
from os.path import isfile, join, isdir
#from PIL import Image
import numpy as np
#import shelve


import matplotlib.pyplot as plt

DIRECTION_REV = False
#DIRECTION_REV = True

class Config:
  
  #DATASET_PATH ="/home/jack/data/highway_14minutes/train10FPS"
  #SINGLE_TEST_PATH = "/home/jack/data/highway_14minutes/test10FPS"

  #DATASET_PATH ="/home/jack/data/trafficCam/trafficFlowImages/train"
  #SINGLE_TEST_PATH = "/home/jack/data/trafficCam/trafficFlowImages/testOdd"

  SINGLE_TEST_PATH = "/home/jack/data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test032"
  DATASET_PATH ="/home/jack/data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
  
  BATCH_SIZE = 4
  EPOCHS = 2# 25 # change back to 3 (Jack)
  MODEL_PATH = "/home/jack/src/video-anomaly-detection-master/notebooks/lstmautoencoder/model.hdf5"
  
  #MODEL_PATH = "/home/jack/src/video-anomaly-detection-master/notebooks/lstmautoencoder/model_lstm.hdf5"



def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The desired distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 256 X 256
    sequence_size: int
        The size of the desired LSTM sequence
    Returns
    -------
    list
        A list of clips , 10 frames each
    """
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips

import cv2
def get_training_set():
    #noiseX, noiseY = 40, 40
    """
    Returns
    -------
    list
        A list of training sequences of shape (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    #####################################
    # cache = shelve.open(Config.CACHE_PATH)
    # return cache["datasetLSTM"]
    #####################################
    clips = []
    # loop over the training folders (Train000,Train001,..)
    for f in sorted(listdir(Config.DATASET_PATH)):
        
        if isdir(join(Config.DATASET_PATH, f)):
            all_frames = []
            # loop over all the images in the folder (0.tif,1.tif,..,199.tif)
            for c in sorted(listdir(join(Config.DATASET_PATH, f))):
              #if str(join(join(Config.DATASET_PATH, f), c))[-3:] == "bmp": #"tif":
              if str(join(join(Config.DATASET_PATH, f), c))[-3:] == "tif":
                    #img = Image.open(join(join(Config.DATASET_PATH, f), c)).resize((256, 256))
                    img = cv2.resize(cv2.imread(join(join(Config.DATASET_PATH, f), c), 0), (256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    #imgplot = plt.imshow(img)# noise test
                    #plt.show()
                    #exit()
                    all_frames.append(img)
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    return clips


def get_single_test():
    #noiseX, noiseY = 40, 40
    sz = 200
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    #print('test path', sorted(listdir(Config.SINGLE_TEST_PATH)))
    #for f in sorted(listdir(Config.SINGLE_TEST_PATH)):
    for f in sorted(listdir(Config.SINGLE_TEST_PATH), reverse=DIRECTION_REV):
        if str(join(Config.SINGLE_TEST_PATH, f))[-3:] == "tif":
        #if str(join(Config.SINGLE_TEST_PATH, f))[-3:] == "bmp":
            #print("true", f)
            
            #img = Image.open(join(Config.SINGLE_TEST_PATH, f)).resize((256, 256))
            img = cv2.resize(cv2.imread(join(Config.SINGLE_TEST_PATH, f), 0), (256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            #img[0:noiseY,0:noiseX] = np.random.rand(noiseY,noiseX) # noise test
            test[cnt, :, :, 0] = img
            cnt = cnt + 1
    return test


#import keras

from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, LayerNormalization
from tensorflow.keras.models import Sequential, load_model

def get_model(reload_model=True):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    if not reload_model:
        return load_model(Config.MODEL_PATH,custom_objects={'LayerNormalization': LayerNormalization})
    training_set = get_training_set()
    #print('training set sizes', len(training_set),len(training_set[0]),len(training_set[0][0]))#,len(training_set[0,0,0,0]),
          #len(training_set[0,0,0,0,0]))
    #exit()
    training_set = np.array(training_set[:600])
    training_set2 = np.array(training_set[600:1200])
    print('training set shape', training_set.shape)
    #print('an image',training_set[0][0])
    training_set = training_set.reshape(-1,10,256,256,1)
    training_set2 = training_set2.reshape(-1,10,256,256,1)
    #exit()
    #training_set = training_set.reshape(500,10,256,256,1)
    
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    #print(seq.summary())
    seq.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

    seq.fit(training_set, training_set,
            batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)
    #seq.fit(training_set2, training_set2,
    #        batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)
    seq.save(Config.MODEL_PATH)
    return seq



def evaluate():
    model = get_model(True)
    print("got model")
    
    test = get_single_test()
    print('test.shape',test.shape)
    sz = test.shape[0] - 10 + 1
    
    sequences = np.zeros((sz, 10, 256, 256, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    print("got data")
    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=4)
    
    #print('predict shape',reconstructed_sequences.shape)
    #print('single value',reconstructed_sequences[0][0][0][0][0])
    imgplot = plt.imshow(reconstructed_sequences[0][0])
    plt.show()
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],
                                                                         reconstructed_sequences[i])) for i in range(0,sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()

evaluate()
