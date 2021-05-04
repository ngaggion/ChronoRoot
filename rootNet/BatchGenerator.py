""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from queue import Queue
from threading import Thread
import time
import numpy as np
import logging
import cv2


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return np.float32(cv2.LUT(image.astype('uint8'), table))


""" #################################################################################################################
                                              BatchGenerator Class
    ################################################################################################################# """

class BatchGenerator:
    """
        This class implements an interface for batch generators. Any batch generator deriving from this class
        should implement its own data augmentation strategy.

        In this class, we suppose that in one epoch we process several subepochs, every one formed by multiple batches
        (every batch is processed independently by the CNN). In every subepoch, new volumes are read from disk
        and used to sample segments for several batches.

        The method generateBatches( ) runs a thread that will call the abstract method generateBatchesForOneEpoch( ) as many
        times as confTrain['numEpochs'] indicates. The method generateBatchesForOneEpoch( ) runs several subepochs,
        reading new volumes in every subepoch. In every subepoch, this method generates several batches composed
        of different random segments extracted from the loaded volumes.
        The batches will be all queued in self.queue using self.queue.put(batch).

        Note that generateBatchesForOneEpoch( ) should read data only once, produce all the necessary batches for the
        given subepochs, insert them at the end of the queue and close the data files.

        The idea is that all the batches (corresponding to all the epochs) are stored (in order) in the same queue.

        The methods that must be implemented by any class inheriting BatchGenerator are:
        - generateBatchesForOneEpoch(self):
            This method will read some data, produce all the necessary batches for the subepochs, insert them at
            the end of the queue and close the data files.

        The class using any BatchGenerator will proceed as follows:

        ===================== Training Loop =====================

            batchGen = BatchGenerator(confTrain)
            batchGen.generateBatches()

            for e in range(0, confTrain['numEpochs']):
                for se in range (0, confTrain['numSubepochs']):
                    batchesPerSubepoch = confTrain['numTrainingSegmentsLoadedOnGpuPerSubep'] // confTrain['batchSizeTraining']

                    for bps in range(0, batchesPerSubepoch)
                        batch = batchGen.getBatch()
                        updateCNN(batch)

            assert (batchGen.emptyQueue()), "The training loop finished before the queue is empty".

        ==========================================

         ====== Pseudocode of the data generation loop =======
         for epoch in range(0, numEpochs):
           for subepoch in range (0, numSubepochs):
               read 'numOfCasesLoadedPerSubepoch' volumes
               batchesPerSubepoch = numberTrainingSegmentsLoadedOnGpuPerSubep // batchSizeTraining
               for batch in range(0, batchesPerSubepoch):
                   data = extractSegments(batchSizeTraining)
                   queue.put(data)

         ====== Pseudocode of the training loop (running in parallel with data generation) =======
         for epoch in range(0, numEpochs):
           for subepoch in range (0, numSubepochs):
               batchesPerSubepoch = numberTrainingSegmentsLoadedOnGpuPerSubep // batchSizeTraining
               for batch in range(0, batchesPerSubepoch):
                   data = queue.get()
                   updateCNNWeights(data)
    """

    def __init__(self, confTrain, confModel, maxQueueSize = 50, infiniteLoop = False):
        """
            Creates a batch generator.

            :param confTrain: a configuration dictionary containing the necessary training parameters
            :param maxQueueSize: maximum number of batches that will be inserted in the queue at the same time.
                           If this number is achieved, the batch generator will wait until one batch is
                           consumed to generate a new one.

                           The number of elements in the queue can be monitored using getNumBatchesInQueue. The queue
                           should never be empty so that the GPU is never idle. Note that the bigger maxQueueSize,
                           the more RAM will the program consume to store the batches in memory. You should find
                           a good balance between RAM consumption and keeping the GPU processing batches all the time.

            :param infiniteLoop: if it's True, then epochs are ignored and the batch generator itearates until it's
                                killed by the system.

            :return: self.queue.empty()
        """
        self.confTrain = confTrain
        self.confModel = confModel
        logging.debug("Creating Queue with size: " + str(maxQueueSize))
        self.queue = Queue(maxsize=maxQueueSize)
        self.currentEpoch = 0
        self.infiniteLoop = infiniteLoop
        self.keepOnRunning = True

        # ID used when printing LOG messages.
        self.id = "[BATCHGEN]"

    def emptyQueue(self):
        """
            Checks if the batch queue is empty or not.

            :return: self.queue.empty()
        """

        return self.queue.empty()

    def _generateBatches(self):
        """
            Private function that generates as many batches as epochs were specified
        """
        self.currentEpoch = 0

        while (self.infiniteLoop or (self.currentEpoch < self.confTrain['numEpochs'])) and self.keepOnRunning:
            self.generateBatchesForOneEpoch()
            self.currentEpoch += 1

        logging.debug(self.id + " The batch generation process finished. Elements still in the queue before finishing: %s. The queue will be destroyed." % str(self.getNumBatchesInQueue()))

    def generateBatches(self):
        """
            This public interface lunches a thread that will start generating batches for the epochs/subepochs specified
            in the configuration file, and storing them in the self.queue.
            To extract these batches, use self.getBatch()
        """
        worker = Thread(target=self._generateBatches, args=())
        worker.setDaemon(True)
        worker.start()

    def getBatch(self):
        """
            It returns a batch and removes it from the front of the queue

            :return: a batch from the queue
        """
        batch = self.queue.get()
        self.queue.task_done()
        return batch

    def getNumBatchesInQueue(self):
        """
            It returns the number of batches currently in the queue, which are ready to be processed

            :return: number of batches in the queue
        """
        return self.queue.qsize()

    def finish(self, delay = .5):
        """
            It will interrupt the batch generation process, even if there are still batches to be created.
            If the batch generator is currently producing a batch, then it will stop after finishing that batch.
            The queue will be destroyed together with the process.

            Note: if there is a process waiting for a batch in the queue, the behaviour is unpredictable. The process
                  that is waiting may wait forever.

            :param delay: the delay will be used after getting an element from the queue. If your batch generation
                        process is too time consuming, you should increase the delay to guarantee that once
                        the queue is empty, the batch generation process is done.
        """
        self.keepOnRunning = False
        logging.debug(self.id + " Stopping batch generator. Cleaning the queue which currently contains %s elements ..." % str(self.queue.qsize()))

        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

            time.sleep(delay)
            if not self.queue.empty():
                logging.debug(self.id + " Still %s elements in the queue ..." % str(self.queue.qsize()))

        logging.debug(self.id + " Done.")

    def generateBatchesForOneEpoch(self):
        """
            This abstract function must be implemented. It must generate all the batches corresponding to one epoch
            (one epoch is divided in subepochs where different data samples are read from disk, and every subepoch is
            composed by several batches, where every batch includes many segments.)

            Every batch must be queued using self.queue.put(batch) and encoded using lasagne-compatible format,
            i.e: a 5D tensor with size (batch_size, num_input_channels, input_depth, input_rows, input_columns)

        """
        raise NotImplementedError('users must define "generateBatches" to use this base class')


""" #################################################################################################################
                                              Simple2DBatchGeneratorFromTensors Class
    ################################################################################################################# """
class Patch2DBatchGeneratorFromTensors(BatchGenerator):
    """
        2D Patch based batch generator from images stored in a Numpy Tensor.
    """

    def __init__(self, confTrain, data, gt, random_state = None, augment = False, indicesToSampleFrom = None, maxQueueSize = 5, infiniteLoop=False):
        """
        It creates a patch based 2D generator

        :param confTrain: configuration dictoriony. It need to define, at least:
            confTrain['numEpochs']: Number of epochs
            confTrain['tileSize']: list with size of the patches to be sampled. "-1" means the complete image in this axis. E.g: [-1, 20] will sample the complete image in X and 20 in Y.
            confTrain['batchSize']: number of patches per batch

        :param data is dictionary indexed by sample ID, where every (numSamples, numChannels, w, h)

        :param maxQueueSize:
        """
        
        if indicesToSampleFrom is None:
            self.indicesToSampleFrom = list(range(len(data)))
        else:
            self.indicesToSampleFrom = indicesToSampleFrom

        BatchGenerator.__init__(self, confTrain, None, maxQueueSize, infiniteLoop=infiniteLoop)
        self.data = data
        self.gt = gt
        self.augment = augment
        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(0)
            
        self.i = 0
        self.n = len(data)
        #print("Len data", self.n)

    def _augment(self, image, label):    
        batchSize = self.confTrain['batchSize']
        
        for i in range(0,batchSize):

            coin = self.random_state.uniform(0,1)
            if coin < 0.5:        
                image[i,:,:,0] = cv2.flip(image[i,:,:,0],1)
                label[i,:,:,:] = cv2.flip(label[i,:,:,:].astype('uint8'),1).astype('bool')
            

            coin = self.random_state.uniform(0,1)
            if coin < 0.75:    
                value = self.random_state.uniform(0.75,1.25)
                image[i,:,:,0] = adjust_gamma(image[i,:,:,0], value)
            
            coin = self.random_state.uniform(0,1)
            if coin < 0.15:
                image[i,:,:,0] = cv2.GaussianBlur(image[i,:,:,0],(5,5),0)
            else:
                if coin < 0.30:
                    image[i,:,:,0] = cv2.medianBlur(image[i,:,:,0].astype('uint8'), 9).astype('float32')

            coin = self.random_state.uniform(0,1)
            if coin < 0.25:
                noise = self.random_state.randn(image.shape[1],image.shape[2]).astype('float32')*5
                image[i,:,:,0] = cv2.add(image[i,:,:,0], noise)
                image[i,:,:,0] = np.clip(image[i,:,:,0], 0, 255)
         
        return image, label
    
    def generateNRandomPatchs(self):
        sampledImg = self.indicesToSampleFrom[self.i]
        
        image = self.data[sampledImg]
        label = self.gt[sampledImg]
               
        imgShape = image.shape[0:2]        
        tileSize = self.confTrain['tileSize']
        
        # Calculate the offsets
        tileOffset = [x // 2 for x in tileSize]
        extraTileOffset = [x % 2 for x in tileSize]

        # Choose the random center from the possible values
        p0 = [x // 2 for x in tileSize]
        p1 = [x - p0[i] - extraTileOffset[i] for i, x in enumerate(imgShape)]
        
        a_center =  np.where(label[p0[0]:p1[0],p0[1]:p1[1],0] == 1)
        l_a = len(a_center[0])
        
        b_center =  np.where(label[p0[0]:p1[0],p0[1]:p1[1],1] == 1)
        l_b = len(b_center[0])
        
        N = self.confTrain['batchSize']
        dataOut = np.ndarray(shape=(N, tileSize[0], tileSize[1], 1), dtype=np.float32)
        gtOut = np.ndarray(shape=(N, tileSize[0], tileSize[1], 2), dtype=np.float32)

        for i in range(0, N):
            coin = self.random_state.uniform(0,1)
            if coin > 0.5:
                if l_a > 1:
                    r = self.random_state.randint(0,l_a-1)
                    samplingCenter = [a_center[0][r]+p0[0],a_center[1][r]+p0[1]]
                else:
                    samplingCenter = [self.random_state.randint(p0[0], p1[0]), 
                                      self.random_state.randint(p0[1], p1[1])]
            else:
                if l_b > 1:
                    r = self.random_state.randint(0,l_b-1)
                    samplingCenter = [b_center[0][r]+p0[0],b_center[1][r]+p0[1]]
                else:
                    samplingCenter = [self.random_state.randint(p0[0], p1[0]), 
                                      self.random_state.randint(p0[1], p1[1])]

            dataOut[i, :, :, 0] = image[samplingCenter[0] - tileOffset[0]:samplingCenter[0] + tileOffset[0] + extraTileOffset[0],
                                        samplingCenter[1] - tileOffset[1]:samplingCenter[1] + tileOffset[1] + extraTileOffset[1]]
    
            gtOut[i, :, :, :] = label[samplingCenter[0] - tileOffset[0]:samplingCenter[0] + tileOffset[0] + extraTileOffset[0],
                                      samplingCenter[1] - tileOffset[1]:samplingCenter[1] + tileOffset[1] + extraTileOffset[1]]

        if self.augment:
            dataOut, gtOut = self._augment(dataOut,gtOut)
            
        dataOut = dataOut/255.0

        return dataOut, gtOut


    def generateSingleBatch(self):

        if self.i == 0:
            #print('List shuffled.')
            self.indicesToSampleFrom = self.random_state.permutation(self.n)
        
        batch, gt = self.generateNRandomPatchs()
        batch = np.clip(batch,0,1)
        gt = np.clip(gt,0,1)

        self.i = (self.i + 1) % self.n

        return batch, gt

    def generateBatchesForOneEpoch(self):
        data, gt = self.generateSingleBatch()
        self.queue.put((data, gt))


class Patch2DBatchGeneratorFromTensors_classic(BatchGenerator):
    """
        2D Patch based batch generator from images stored in a Numpy Tensor.
    """

    def __init__(self, confTrain, data, gt, random_state = None, indicesToSampleFrom = None, maxQueueSize = 5, infiniteLoop=False):
        """
        It creates a patch based 2D generator

        :param confTrain: configuration dictoriony. It need to define, at least:
            confTrain['numEpochs']: Number of epochs
            confTrain['tileSize']: list with size of the patches to be sampled. "-1" means the complete image in this axis. E.g: [-1, 20] will sample the complete image in X and 20 in Y.
            confTrain['batchSize']: number of patches per batch

        :param data is dictionary indexed by sample ID, where every (numSamples, numChannels, w, h)

        :param maxQueueSize:
        """
        if indicesToSampleFrom is None:
            self.indicesToSampleFrom = list(range(len(data)))
        else:
            self.indicesToSampleFrom = indicesToSampleFrom

        BatchGenerator.__init__(self, confTrain, None, maxQueueSize, infiniteLoop=infiniteLoop)
        self.data = data
        self.gt = gt
        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(0)

    def generateRandomPatch(self):
        sampledImg = self.random_state.choice(self.indicesToSampleFrom)
        imgShape = self.data[sampledImg].shape[0:2]

        tileSize = self.confTrain['tileSize']
        # Replace -1 by the current image size
        # tileSize = [x if x != -1 else imgShape[i] for i, x in enumerate(tileSize)]

        # Calculate the offsets
        tileOffset = [x // 2 for x in tileSize]
        extraTileOffset = [x % 2 for x in tileSize]

        # Choose the random center from the possible values
        p0 = [x // 2 for x in tileSize]
        p1 = [x - p0[i] - extraTileOffset[i] for i, x in enumerate(imgShape)]
        
        coin = self.random_state.uniform(0,1)
        if coin < 0.5:
            a =  np.where(self.gt[sampledImg][p0[0]:p1[0],p0[1]:p1[1],0] == 1)
            l = len(a[0])
            if l > 1:
                r = self.random_state.randint(0,l-1)
                samplingCenter = [a[0][r]+p0[0],a[1][r]+p0[1]]
            else:
                samplingCenter = [self.random_state.randint(p0[0], p1[0]), 
                                  self.random_state.randint(p0[1], p1[1])]
        else:
            a =  np.where(self.gt[sampledImg][p0[0]:p1[0],p0[1]:p1[1],1] == 1)
            l = len(a[0])
            if l > 1:
                r = self.random_state.randint(0,l-1)
                samplingCenter = [a[0][r]+p0[0],a[1][r]+p0[1]]
            else:
                samplingCenter = [self.random_state.randint(p0[0], p1[0]), 
                                  self.random_state.randint(p0[1], p1[1])]

        dataOut = np.ndarray(shape=(1, tileSize[0], tileSize[1], 1),
                      dtype=np.float32)
        gtOut = np.ndarray(shape=(1, tileSize[0], tileSize[1], 2),
                    dtype=np.float32)


        dataOut[0, :, :, 0] = self.data[sampledImg][samplingCenter[0] - tileOffset[0]:samplingCenter[0] + tileOffset[0] + extraTileOffset[0],
                                                    samplingCenter[1] - tileOffset[1]:samplingCenter[1] + tileOffset[1] + extraTileOffset[1]]/255.0

        gtOut[0, :, :, :] = self.gt[sampledImg][samplingCenter[0] - tileOffset[0]:samplingCenter[0] + tileOffset[0] + extraTileOffset[0],
                                                samplingCenter[1] - tileOffset[1]:samplingCenter[1] + tileOffset[1] + extraTileOffset[1]]

        #print "Patch: Center " + str(samplingCenter) + "    Size: " + str(tileSize)
        return dataOut, gtOut

    def generateSingleBatch(self):
        """
            Creates a batch of segments according to the conf file. It supposes that the images are already
            loaded in self.currentVolumes.

            :return: It returns the data and ground truth of a complete batch as data, gt. These structures are theano-compatible with shape:
                        np.ndarray(shape=(self.confTrain['batchSizeTraining'], self.numChannels, tileSize, tileSize, tileSize), dtype=np.float32)
        """
        
        batchSize = self.confTrain['batchSize']

        batch = np.ndarray(shape=(batchSize, self.confTrain['tileSize'][0], self.confTrain['tileSize'][1], 1), dtype=np.float32)
        gt =    np.ndarray(shape=(batchSize, self.confTrain['tileSize'][0], self.confTrain['tileSize'][1], 2), dtype=np.float32)

        for i in range(0, batchSize):
            batch[i, :, :, :], gt[i, :, :, :] = self.generateRandomPatch()
	
        batch = np.clip(batch,0,1)
        gt = np.clip(gt,0,1)
        return batch, gt

    def generateBatchesForOneEpoch(self):
        data, gt = self.generateSingleBatch()
        self.queue.put((data, gt))
