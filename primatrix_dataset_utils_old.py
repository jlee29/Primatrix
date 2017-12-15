from itertools import cycle
import numpy as np
import os
import pandas as pd
import skvideo.io as skv

from warnings import filterwarnings
filterwarnings('ignore')

from multilabel import multilabel_train_test_split


class Dataset(object):
    
    def __init__(self, datapath, dataset_type='micro', reduce_frames=False, val_size=0.3, batch_size=16, test=False):
        
        self.datapath = datapath
        self.dataset_type = dataset_type
        self.reduce_frames = reduce_frames
        self.val_size = val_size
        self.batch_size = batch_size
        
        # boolean for test mode
        self.test = test
        
        # params based on dataset type
        if self.dataset_type == 'nano':
            self.height = 16
            self.width = 16
        elif self.dataset_type == 'micro':
            self.height = 64
            self.width = 64
        elif self.dataset_type == 'raw':
            print("\nRaw videos have variable size... \nsetting height and width to None... \nfirst video in test will determine size (test must be True ")
            self.height = None
            self.weidth = None
        else:
            raise NotImplementedError("Please set dataset_type as raw, micro, or nano.")
            
        # params based on frame reduction
        if self.reduce_frames:
            self.num_frames = 15
        else:
            self.num_frames = 30
        
        
        # training and validation        
        self.X_train, self.X_val, self.y_train, self.y_val = self.split_training_into_validation()
        self.num_train = self.y_train.shape[0]
        self.num_val = self.y_val.shape[0]
    
        # params of data based on training data
        self.num_classes = self.y_train.shape[1]
        #self.class_names = self.y_train.columns.values
        self.num_samples = self.y_train.shape[0]
        self.num_batches = self.num_samples // self.batch_size
        
        # test paths and prediction matrix
        self.pred_files_all, self.pred_files_with_data, self.pred_probs_with_data = self.prepare_test_data_and_prediction()
        self.num_test_all = len(self.pred_files_all)
        self.num_test = len(self.pred_files_with_data)
        
        
        # variables to make batch generating easier
        #self.batch_idx = cycle(range(self.num_batches))
        #self.batch_num = next(self.batch_idx)
        
        self.num_val_batches = self.y_val.shape[0] // self.batch_size
        self.val_batch_idx = cycle(range(self.num_val_batches))
        self.val_batch_num = next(self.val_batch_idx)
        
        #self.num_test_samples = self.X_test_ids.shape[0]
        #self.num_test_batches = self.num_test_samples // self.batch_size
        #self.test_batch_idx = cycle(range(self.num_test_batches))
        #self.test_batch_num = next(self.test_batch_idx)
        
    
        # for testing iterator in test_mode
        #self.train_data_seen = pd.DataFrame(data={'seen': 0}, index=self.y_train.index)
        
        # test the generator
        #if test:
        #    self._test_batch_generator()
    
    def prepare_test_data_and_prediction(self):
        """
        Returns paths to test data indexed by subject_id 
        and preallocates prediction dataframe.
        """
        datapath = self.datapath
        dataset_type = self.dataset_type
        subjpath = os.path.join(datapath, dataset_type)
        avail_data = set(os.listdir(subjpath))
        predpath = os.path.join(self.datapath, 'submission_format.csv')
        pred_files_all = [line.strip().split(',')[0] for line in open(predpath)][1:]
        pred_files_with_data = [line.strip().split(',')[0] for line in open(predpath) if line.strip().split(',')[0] +'.npy' in avail_data]        
        pred_probs_all = np.zeros([len(pred_files_all), self.num_classes])
        pred_probs_with_data = np.zeros([len(pred_files_with_data), self.num_classes])
        return pred_files_all, pred_files_with_data, pred_probs_with_data
  
    
    def split_training_into_validation(self):
        """
        Uses the multilabel_train_test_split function 
        to maintain class distributions between train
        and validation sets.
        """        
        datapath = self.datapath
        dataset_type = self.dataset_type
        val_size = self.val_size
        
        subjpath = os.path.join(datapath, dataset_type)
        avail_data = set(os.listdir(subjpath))
        labelpath = os.path.join(datapath, 'train_labels.csv')
        labeled_files = {line.strip().split(',')[0] : line.strip().split(',')[1:] for line in open(labelpath) if line.strip().split(',')[0] +'.npy' in avail_data}
        X = []
        y = []
        for labeled_file in labeled_files:
            line = labeled_files[labeled_file]
            for i in range(len(line)):
                line[i] = float(line[i])
            labeled_files[labeled_file] = line
            y.append(line)
            X.append(os.path.join(subjpath, labeled_file) + '.npy')
        X = np.array(X)
        y = np.array(y)
        arr = np.arange(len(X))
        np.random.shuffle(arr)
        num_val = int(val_size*len(X))
        X_val = X[arr[:num_val]]
        y_val = y[arr[:num_val]]            
        X_train = X[arr[num_val:]]
        y_train = y[arr[num_val:]]
        return X_train, X_val, y_train, y_val
    
    def data_augment(self, x):
        """
        Stochastically augments the single piece of data.
        INPUT:
        - data_iter: (3d ND-array) the single piece of data
        - data_seg: (2d ND-array) the corresponding segmentation
        """
        matrix_size = x.shape[1]
        # Creating Random variables
        roller = np.round(float(matrix_size/8))
        ox, oy = np.random.randint(-roller, roller+1, 2)
        do_flip = np.random.randn() > 0
        pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
        add_rand = np.clip(np.random.randn() * 0.1, -.4, .4)
        # Rolling
        x = np.roll(np.roll(x, ox, 0), oy, 1)
        # Left-right Flipping
        if do_flip:
            x = np.fliplr(x)
        # Raising/Lowering to a power
        x = x ** pow_rand
        # Random adding of shade.
        x += add_rand
        return x
        
    def batches(self, verbose=False):
        """This method yields the next batch of videos for training."""
        num_train = self.y_train.shape[0]
        batch_size = self.batch_size
        
        x = np.zeros([batch_size, self.num_frames, self.height, self.width, 1])
        y = np.zeros([batch_size, self.num_classes])
        while 1:
            ind_list = np.random.choice(range(num_train), batch_size, replace=True)
            for i in range(batch_size):
                ind = ind_list[i]
                video = self.data_augment(np.load(self.X_train[ind]).astype(np.float32))
                #video = np.load(self.X_train[ind]).astype(np.float32)goog
                x[i] = video
                y[i] = self.y_train[ind]        
            yield (x, y)
            
            
    def val_batches(self, verbose=False):
        """This method yields the next batch of videos for validation."""
        batch_size = self.batch_size
        num_train = self.y_train.shape[0]        
        x = np.zeros([batch_size, self.num_frames, self.height, self.width, 1])
        y = np.zeros([batch_size, self.num_classes])
        while 1:
            start = self.batch_size*self.val_batch_num
            #stop = self.batch_size*(self.val_batch_num + 1)
            for i in range(batch_size):
                ind = start + i
                video = np.load(self.X_val[ind]).astype(np.float32)
                x[i] = video
                y[i] = self.y_val[ind]        
            yield (x, y)
            
    def test_batches(self, start, verbose=False):
        """This method yields the next batch of videos for testing."""
        datapath = self.datapath
        dataset_type = self.dataset_type
        
        subjpath = os.path.join(datapath, dataset_type)
        
        batch_size = min(self.batch_size, self.num_test - start)          
        x = np.zeros([batch_size, self.num_frames, self.height, self.width, 1])
        for i in range(batch_size):
            ind = start + i
            video = np.load(os.path.join(subjpath, self.pred_files_with_data[ind] + '.npy')).astype(np.float32)
            x[i] = video
        return x

        
    def update_predictions(self, results, start):
        batch_size = min(self.batch_size, self.num_test - start)   
        self.pred_probs_with_data[start:start+batch_size] = results
        
    def write_csv(self, predpath):
        avg = np.mean(self.pred_probs_with_data, axis=0)
        file_to_index = {self.pred_files_with_data[i] : i for i in range(self.num_test)}
        formatpath = os.path.join(self.datapath, 'submission_format.csv')
        header = [line for line in open(formatpath)][0]
        f = open(predpath, 'w')
        f.write(header)
        for i in range(self.num_test_all):
            filename = self.pred_files_all[i]
            line = filename + ','
            if filename in file_to_index:
                pred_probs = self.pred_probs_with_data[file_to_index[filename]].astype(str)
            else:
                pred_probs = avg.astype(str)
            line += ','.join(pred_probs)
            line += '\n'
            f.write(line)
        f.close()
        