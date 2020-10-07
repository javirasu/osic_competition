import pydicom
import numpy as np
import glob
import os
from tqdm import tqdm
import yaml
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from skimage.transform import resize


class label_encoder:
    def __init__(self, start_id=0):
        self.encoder_ = dict()
        self.decoder_ = dict()
        self.next_ = start_id

    def fit(self, vals):
        for x in vals:
            if self.encoder_.get(x, None) == None:
                self.encoder_[x] = self.next_
                self.next_ += 1

        self.decoder_ = {value : key for (key, value) in self.encoder_.items()}

    def transform(self, vals):
        if len(vals) > 1:
            return [self.encoder_[x] for x in vals]
        else:
            return self.encoder_[vals[0]]
    
    def inverse_transform(self, vals):
        if len(vals) > 1:
            return [self.decoder_[x] for x in vals]
        else:
            return self.decoder_[vals[0]]

def extract_img_id(x):
    string = x.replace('\\','/')
    string = string.split('/')[-1]
    string = string.split('.')[0]
    return int(string)

def normalize_img(x, x_min, x_max):
    return ((x-x_min)/((x_max-x_min))).astype(np.float32)

def load_ct_scans(path_imgs, patient_ids, n_frames=30, frames_size=(224,224)):

    if not path_imgs.endswith('/'):
        path_imgs +='/'

    data = dict()

    # Load image
    problem_ct = []

    for patient in tqdm(patient_ids):

        image_files = glob.glob(os.path.join(path_imgs+patient,'*.dcm'))
        image_order = np.argsort(np.array([extract_img_id(x) for x in image_files]))

        imgs_list = []

        for i in image_order:
            image_data = pydicom.read_file(image_files[i])
            patient_id = image_data.PatientID
            if patient_id != patient:
                raise Exception('Patient Ids do not match')  
            
            try:
                img = np.array(image_data.pixel_array)
            except:
                problem_ct.append(patient)

            
            # Removing empty space around the CT
            mask_rows = np.argwhere(np.sum(img, axis=1) != 0).squeeze()
            mask_cols = np.argwhere(np.sum(img, axis=0) != 0).squeeze()
            img = img[mask_rows,:][:,mask_cols]
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
            imgs_list.append(img)
            
        if len(imgs_list) == 0:
            continue

        # Subsampling to 30 frames
        if len(imgs_list) > n_frames:
            subsample = np.linspace(0, len(imgs_list)-1, num=n_frames, dtype=np.int32).tolist()
            imgs_list = [imgs_list[i] for i in subsample]

        # Making all images same dimensions
        imgs_list = [resize(x, frames_size, anti_aliasing=True) for x in imgs_list]

        # Normalizing frames
        min_val = int(1e6)
        max_val = int(-1e6)
        for img in imgs_list:
            min_x = np.min(img)
            max_x = np.max(img)
            if min_x < min_val:
                min_val = min_x
            if max_x > max_val:
                max_val = max_x

        imgs_list = [normalize_img(x, min_val, max_val) for x in imgs_list]
        
        # Fill up in case there are less than 30 frames        
        if len(imgs_list) < n_frames:
            n_diff = n_frames - len(imgs_list)
            black_frame = np.zeros((frames_size[0], frames_size[1]), dtype=np.float32)
            for _ in range(n_diff):
                imgs_list.insert(0, black_frame)

        frames_scan = np.stack(imgs_list,0)
        if np.min(frames_scan) != 0 or np.max(frames_scan) != 1:
            raise Exception('Error normalization patient '+str(patient_encoder.transform([patient]))) 

        # Repeat to 3 channels for vgg
        frames_scan = np.expand_dims(frames_scan, -1)

        #Store data
        data[patient] = frames_scan.astype(np.float32)

    # Filter to remove background
    for k,v in data.items():
        ## Preprocess to get into VGG preprocessing
        data[k] = filter_by_std(v, alpha=0.01)

    return data, np.unique(problem_ct)

def process_tabula(df):

    le_patient = label_encoder()
    le_patient.fit(np.unique(df.Patient).tolist())
    sex_encoding = {'Male':0, 'Female':1}
    status_encoding = {'Currently smokes': np.array([0,0,1]), 'Ex-smoker': np.array([0,1,0]), 'Never smoked': np.array([1,0,0])}

    df.Patient = df.Patient.apply(lambda x: le_patient.transform([x]))
    df.Sex = df.Sex.apply(lambda x: sex_encoding[x])
    df.SmokingStatus = df.SmokingStatus.apply(lambda x: status_encoding[x])

def competition_metric(fvc_true, fvc_pred, fvc_std):
    std_clipped = np.maximum(fvc_std, 70)
    error = np.minimum(np.abs(fvc_true - fvc_pred), 1000)
    metric = - np.sqrt(2) * error/std_clipped - np.log(np.sqrt(2) * std_clipped)

    return metric

def get_config(path=r'../config.yaml'):
    with open(path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def make_ct_git(data, filename):
    
    fig = plt.figure()

    ims = []
    for image in data:
        im = plt.imshow(image.squeeze(), animated=True, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("off")
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)
    ani.save(filename, writer='imagemagick', fps=10)

def filter_by_std(x, alpha=0.01):
    # Remove top 1% pixel intensity
    q_pixel_max = np.quantile(x, 0.99, interpolation='linear')
    x = np.clip(x, 0, q_pixel_max)

    # Remove background just where std is below min
    mu_mat = np.mean(x, axis=0)
    std_mat = np.std(x, axis=0)
    q_std_min = np.quantile(std_mat, 0.10, interpolation='linear')

    x[:,std_mat<q_std_min] = (x-mu_mat)[:,std_mat<q_std_min]

    # ---
    x = (x-x.min())/(x.max()-x.min())
    q_std_minmin = np.quantile(x, 0.1, interpolation='linear')
    x = np.clip(x, q_std_minmin, 1.0)

    # Renormalize
    x = (x-x.min())/(x.max()-x.min())
    return x