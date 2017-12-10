from itertools import cycle
import numpy as np
import os
import pandas as pd
import skvideo.io as skv

from warnings import filterwarnings
filterwarnings('ignore')

from multilabel import multilabel_train_test_split


class Dataset(object):
    
    def __init__(self, datapath, dataset_type='nano', reduce_frames=True, val_size=0.3, batch_size=16, test=False):
        
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
        
        # for tracking errors
        self.bad_videos = []
        
        # training and validation
        self.X_train, self.X_val, self.y_train, self.y_val = self.split_training_into_validation()
    
        # params of data based on training data
        self.num_classes = self.y_train.shape[1]
        self.class_names = self.y_train.columns.values
        assert self.num_classes == self.y_val.shape[1]
        self.num_samples = self.y_train.shape[0]
        self.num_batches = self.num_samples // self.batch_size
        
        # test paths and prediction matrix
        self.X_test_ids, self.predictions = self.prepare_test_data_and_prediction()
        
        # variables to make batch generating easier
        self.batch_idx = cycle(range(self.num_batches))
        self.batch_num = next(self.batch_idx)
        
        self.num_val_batches = self.y_val.shape[0] // self.batch_size
        self.val_batch_idx = cycle(range(self.num_val_batches))
        self.val_batch_num = next(self.val_batch_idx)
        
        self.num_test_samples = self.X_test_ids.shape[0]
        self.num_test_batches = self.num_test_samples // self.batch_size
        self.test_batch_idx = cycle(range(self.num_test_batches))
        self.test_batch_num = next(self.test_batch_idx)
    
        # for testing iterator in test_mode
        self.train_data_seen = pd.DataFrame(data={'seen': 0}, index=self.y_train.index)
        
        # test the generator
        #if test:
        #    self._test_batch_generator()
    
    def prepare_test_data_and_prediction(self):
        """
        Returns paths to test data indexed by subject_id 
        and preallocates prediction dataframe.
        """
        
        predpath = os.path.join(self.datapath, 'submission_format.csv')
        predictions = pd.read_csv(predpath, index_col='filename')
        test_idx = predictions.index
        subjpath = os.path.join(self.datapath, self.dataset_type)
        #subject_ids = pd.read_csv(subjpath, index_col=0)
        subject_ids = pd.DataFrame(data=subjpath, columns=['filepath'], index=test_idx)
        for row in subject_ids.itertuples():
            subject_ids.loc[row.Index] = os.path.join(row.filepath, row.Index) 
        
        return test_idx, predictions
  
    
    def split_training_into_validation(self):
        """
        Uses the multilabel_train_test_split function 
        to maintain class distributions between train
        and validation sets.
        """

        datapath = self.datapath
        dataset_type = self.dataset_type
        val_size = self.val_size
        
        # load training labels
        labelpath = os.path.join(datapath, 'train_labels.csv')
        labels = pd.read_csv(labelpath, index_col='filename')
        
        # load subject labels (assumed to have same index as training labels)
        subjpath = os.path.join(datapath, dataset_type)
        #subject_ids = pd.read_csv(subjpath, index_col=0)
        subject_ids = pd.DataFrame(data=subjpath, columns=['filepath'], index=labels.index)
        for row in subject_ids.itertuples():
            subject_ids.loc[row.Index] = os.path.join(row.filepath, row.Index)       
        
        # split
        X_train, X_val, y_train, y_val = multilabel_train_test_split(subject_ids, labels, size=val_size, min_count=1, seed=0)
        
        # check distribution is maintained
        dist_diff = (y_train.sum()/y_train.shape[0] - y_val.sum() / y_val.shape[0]).sum()
        #print(dist_diff)
        assert np.isclose(dist_diff, 0, rtol=1e-04, atol=1e-02)
        
        return X_train, X_val, y_train, y_val
        
    def batches(self, verbose=False):
        """This method yields the next batch of videos for training."""
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_train = self.y_train.shape[0]
        
        
        
        while 1:
            # get videos
            #start = self.batch_size*self.batch_num
            #stop = self.batch_size*(self.batch_num + 1)
            batch_indices = np.random.choice(range(num_train), batch_size, replace=True)
            
            # print batch ranges if testing
            #if self.test:
            #    print("batch {self.batch_num}:\t{start} --> {stop-1}")
            
            x_paths = self.X_train.iloc[batch_indices]
            x, failed = self._get_video_batch(x_paths, 
                                              True,
                                              reduce_frames=reduce_frames, 
                                              verbose=verbose)
            x_paths = x_paths.drop(failed)
            self.bad_videos += failed

            # get labels
            y = self.y_train.iloc[batch_indices]
            y = y.drop(failed)

            # check match for labels and videos
            assert (x_paths.index==y.index).all()
            assert x.shape[0] == y.shape[0]

            # report failures if verbose
            if len(failed) != 0 and verbose==True:
                print("\t\t\t*** ERROR FETCHING BATCH {self.batch_num}/{self.num_batches} ***")
                print("Dropped {len(failed)} videos:")
                for failure in failed:
                    print("\t{failure}\n\n")

            # increment batch number
            self.batch_num = next(self.batch_idx)
            
            # update dataframe of seen training indices for testing
            self.train_data_seen.loc[y.index.values] = 1
            yield (x, y)
            
            
    def val_batches(self, verbose=False):
        """This method yields the next batch of videos for validation."""
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_val = self.y_val.shape[0]
        
        
        
        while 1:
            # get videos
            batch_indices = np.random.choice(range(num_val), batch_size, replace=False)
            #start = self.batch_size*self.val_batch_num
            #stop = self.batch_size*(self.val_batch_num + 1)
            
            #x_paths = self.X_val.iloc[start:stop]
            x_paths = self.X_val.iloc[batch_indices]
            x, failed = self._get_video_batch(x_paths,
                                              False,
                                              reduce_frames=reduce_frames, 
                                              verbose=verbose)
            x_paths = x_paths.drop(failed)
            self.bad_videos += failed

            # get labels
            #y = self.y_val.iloc[start:stop]
            y = self.y_val.iloc[batch_indices]
            y = y.drop(failed)

            # check match for labels and videos
            assert (x_paths.index==y.index).all()
            assert x.shape[0] == y.shape[0]

            # report failures if verbose
            if len(failed) != 0 and verbose==True:
                print("\t\t\t*** ERROR FETCHING BATCH {self.batch_num}/{self.num_batches} ***")
                print("Dropped {len(failed)} videos:")
                for failure in failed:
                    print("\t{failure}\n\n")

            # increment batch number
            self.val_batch_num = next(self.val_batch_idx)
            
            yield (x, y)
            
    def test_batches(self, verbose=False):
        """This method yields the next batch of videos for testing."""
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_test = self.num_test_samples
        
        test_dir = os.path.join(self.datapath, self.dataset_type)
        
        
        while 1:
            # get videos
            start = self.batch_size*self.test_batch_num
            stop = self.batch_size*(self.test_batch_num + 1)
            
            x_ids = self.X_test_ids[start:stop]
            x_paths = pd.DataFrame(data=[os.path.join(test_dir, filename) for filename in x_ids], 
                                   columns=['filepath'],
                                   index=x_ids)
            #print(x_paths)
            x, failed = self._get_video_batch(x_paths,
                                              False,
                                              reduce_frames=reduce_frames, 
                                              verbose=verbose)
            
            self.test_batch_ids = x_ids.values

            # increment batch number
            self.test_batch_num = next(self.test_batch_idx)
            
            yield x

        
    def _get_video_batch(self, x_paths, is_training, as_grey=True, reduce_frames=True, verbose=False):
        """
        Returns ndarray of shape (batch_size, num_frames, width, height, channels).
        If as_grey, then channels dimension is squeezed out.
        """

        videos = []
        failed = []
        
        for row in x_paths.itertuples():
            filepath = row.filepath
            obf_id = row.Index
            
            # load
            video = skv.vread(filepath, as_grey=as_grey)
            
            # fill video if neccessary
            if video.shape[0] < self.num_frames:
                video = self._fill_video(video) 
            
            # reduce
            if reduce_frames:
                frames = np.arange(0, video.shape[0], 2)
                try:
                    video = video[frames, :, :] #.squeeze() 
                    videos.append(self.augment(video))
                    #print(video.shape)
                    #videos.append(video)
                
                except IndexError:
                    if verbose:
                        print("FAILED TO REDUCE: {filepath}")
                    print("id:\t{obf_id}")
                    failed.append(obf_id)
            else:
                videos.append(self.augment(video))                       
        return np.array(videos), failed
    
    def _fill_video(self, video):
        """Returns a video with self.num_frames given at least one frame."""

        # establish boundaries
        target_num_frames = self.num_frames
        num_to_fill = target_num_frames - video.shape[0]

        # preallocate array for filler
        filler_frames = np.zeros(shape=(num_to_fill, self.width, self.height, 1)) # assumes grey

        # fill frames
        source_frame = cycle(np.arange(0, video.shape[0]))
        for i in range(num_to_fill):
            filler_frames[i, :, :] = video[next(source_frame), :, :]

        return np.concatenate((video, filler_frames), axis=0)
    
    def augment(self, x):
        #matrix_size = x.shape[1]
        # Creating Random variables
        #roller = np.round(float(matrix_size/8))
        #ox, oy = np.random.randint(-roller, roller+1, 2)
        #print(x.shape)
        do_flip = np.random.randn() > 0
        pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
        add_rand = np.clip(np.random.randn() * 16., -64., 64.)
        # Rolling
        #x = np.roll(np.roll(x, ox, 0), oy, 1)
        # Left-right Flipping
        if do_flip:
            x = np.transpose(np.fliplr(np.transpose(x, [1,2,0,3])), [2,0,1,3])
        # Raising/Lowering to a power
        x = x ** pow_rand
        # Random adding of shade.
        x += add_rand
        return x
    

    def _test_batch_generator(self):
        
        print('Testing train batch generation...')
        
        for i in range(self.num_batches):
            if self.batch_num % 10 == 0:
                print("\n\t\t\tBATCH \t{self.batch_num}/{self.num_batches}\n")
            
            batch = self.batches(verbose=True)
            x,y = next(batch)
        
            # same batches for videos and labels
            assert x.shape[0] == y.shape[0]
            
            # square videos
            assert x.shape[2] == x.shape[3]
            
            # black and white
            assert x.shape[4] == 1
            
        
        # assert we've seen all data up to remainder of a batch
        assert (self.y_train.shape[0] - self.train_data_seen.sum().values[0]) < self.batch_size
        
        # check that batch_num is reset
        assert self.batch_num == 0
        
        # turn off test mode
        if self.test == True:
            self.test = False
        
        print('Test passed.')
        
    def update_predictions(self, results):
        self.predictions.loc[self.test_batch_ids] = results
        