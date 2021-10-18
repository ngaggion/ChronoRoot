import tensorflow as tf
import logging

from .modelUtils import pixel_wise_softmax, dice_coe_c1
from .unetModels import ResUNet, UNet, ResUNetDS
from .SegNet import SegNet
from .DeepLab import DeepLab

class RootNet(object):
    def __init__(self, sess, config, name, isTrain):
        """
        :param sess:
        :param config:
        :param name:
        :param isTrain:
        """
        
        self.sess = sess
        self.name = name
        self.isTrain = isTrain
        self.summary = None
        self.summaryComponents = []

        if "finetuneLayers" in config:
            self.finetuneLayers = config["finetuneLayers"]
        else:
            self.finetuneLayers = None

        logging.debug("Constructing RootNet Model")

        imShape = [config['batchSize']] + config['tileSize'] + [1]
        logging.debug("Batch Shape:")
        logging.debug(imShape)
        gtShape = [config['batchSize']] + config['tileSize'] + [2]

        # If phase = True, it is training phase. Otherwise is testing. It is used for batch_norm.
        self.phase = tf.compat.v1.placeholder(tf.bool, name='phase')

        self.dropout = config['dropout']
        logging.debug("Dropout")
        logging.debug(self.dropout)

        logging.debug("Model:")

        # Shared part of the network across multiple levels
        if config['Model'] == "ResUNet":
            self.unet = ResUNet("ResUNet", self.finetuneLayers, self.dropout)
            logging.debug("ResUNet")
        elif config['Model'] == "UNet":
            self.unet = UNet("UNet", self.finetuneLayers, self.dropout)
            logging.debug("UNet")
        elif config['Model'] == "ResUNetDS":
            self.unet = ResUNetDS("ResUNetDS", self.finetuneLayers, self.dropout)
            logging.debug("ResUNetDS")
        elif config['Model'] == "SegNet":
            self.unet = SegNet("SegNet", self.finetuneLayers, self.dropout)
            logging.debug("SegNet")
        elif config['Model'] == "DeepLab":
            self.unet = DeepLab("DeepLab", self.finetuneLayers, self.dropout)
            logging.debug("DeepLab")
        else:
            logging.debug("NO MODEL")
            raise Exception("NO MODEL")

        # Input image
        self.x = tf.compat.v1.placeholder(tf.float32, imShape, name='x')

        # GT Image
        self.y = tf.compat.v1.placeholder(tf.float32, gtShape, name='y')

        #self.learning_rate = config['learning_rate']
        # logging.debug("Learning Rate")
        # logging.debug(self.learning_rate)
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        if config['Model'] == "ResUNetDS" or config['Model'] == "ResUNetDS2":
            self.output, self.m_logits = self.unet(self.x, isTrain=self.phase)
        else:
            self.output = self.unet(self.x, isTrain=self.phase)
        
        self.logits = pixel_wise_softmax(self.output)

        if self.isTrain:
            regularizer = tf.add_n([tf.nn.l2_loss(v) for v in self.unet.varList if 'bias' not in v.name])

            if config['loss'] == "cross_entropy":
                self.loss = -tf.reduce_mean(self.y*tf.math.log(tf.clip_by_value(self.logits,1e-10,1.0)), name="cross_entropy")
            else:
                self.soft_dice = (1 - dice_coe_c1(self.logits, self.y))
                self.loss = self.soft_dice
            
            if config['Model'] == "ResUNetDS" or config['Model'] == "ResUNetDS2" :
                self.m_loss = -tf.reduce_mean(self.y*tf.math.log(tf.clip_by_value(self.m_logits,1e-10,1.0)), name="cross_entropy_b")
                self.loss = config['lambda2'] * self.loss +  config['lambda1'] * self.m_loss
            
            
            self.loss_f = self.loss + config["l2"] * regularizer
            
            #self.auc = tf.compat.v1.metrics.auc(self.y[:,:,:,1], self.logits[:,:,:,1], summation_method='careful_interpolation')
            #self.precision = tf.compat.v1.metrics.precision_at_thresholds(self.y[:,:,:,1], self.logits[:,:,:,1],[0.5])
            #self.recall = tf.compat.v1.metrics.recall_at_thresholds(self.y[:,:,:,1], self.logits[:,:,:,1],[0.5])

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                optim = tf.compat.v1.train.AdamOptimizer(name="optim", learning_rate=self.learning_rate)
                self.train = optim.minimize(self.loss_f, var_list=self.unet.varList)

        self.sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))

        if len(self.summaryComponents) > 1:
            self.summary = tf.compat.v1.summary.merge(self.summaryComponents)


    def fit(self, batchX, batchY, learning_rate, summary=None, phase=1):
        if summary is None:
            _, loss= self.sess.run([self.train, self.loss], {self.x: batchX, self.y: batchY, self.phase: phase, self.learning_rate: learning_rate})
        else:
            log, _, loss= self.sess.run([summary, self.train, self.loss], {self.x: batchX, self.y: batchY, self.phase: phase, self.learning_rate: learning_rate})

        if summary is None:
            return loss
        else:
            return log, loss

    def deploy(self, batchX, batchY, phase=0):
        loss, dice, auc, prec, rec = self.sess.run([self.loss, self.hard_dice, self.auc, self.precision, self.recall], {self.x: batchX, self.y: batchY, self.phase: phase})
        return loss, dice, auc, prec, rec

    def segment(self, batchX):
        segmented= self.sess.run(self.logits, {self.x: batchX, self.phase: 0})
        return segmented

    def save(self, dir_path):
        self.unet.save(self.sess, dir_path + "/model.ckpt")

    def restore(self, dir_path):
        self.unet.restore(self.sess, dir_path + "/model.ckpt")
