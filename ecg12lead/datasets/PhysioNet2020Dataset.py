import os
import glob
import random

import numpy as np

import torch
from torch.utils.data import Dataset

from scipy.io import loadmat
from scipy import signal

import copy
from collections import OrderedDict

import h5py


class PhysioNet2020Dataset(Dataset):
    def __init__(self, datasets_dir,use_datasets,use_labels,Fs=400, N_samples=4096 ):
        
        self.datasets_dir = datasets_dir
        self.use_datasets = sorted(use_datasets)
        self.use_labels   = sorted(use_labels)
        self.Fs = Fs
        self.N_samples = N_samples
        
        self.original_Fs = {'CPSC': 500, 'CPSC-Extra': 500, 'StPetersburg': 257,
                            'PTB': 1000, 'PTB-XL': 500, 'Georgia': 500}
        
        self.dataset_paths = {
            'CPSC': os.path.join(self.datasets_dir,'CPSC'),
            'CPSC-Extra': os.path.join(self.datasets_dir,'CPSC-Extra'),
            'StPetersburg':os.path.join(self.datasets_dir,'StPetersburg'),
            'PTB':os.path.join(self.datasets_dir,'PTB'),
            'PTB-XL': os.path.join(self.datasets_dir,'PTB-XL'),
            'Georgia': os.path.join(self.datasets_dir,'Georgia')
        }

        self.dx_map_decode,self.dx_map_encode,self.y_encode, self.y_decode = self.get_mappings()
        
        self.recordings_index = []
            
        self.prepare_dataset()
        self.setup()
        
        
        
    def get_mappings(self):
        
        #DxMap
        csv_content = """
            Dx,SNOMED CT Code,Abbreviation
            1st degree av block,270492004,IAVB
            2nd degree av block,195042002,IIAVB
            abnormal QRS,164951009,abQRS
            accelerated junctional rhythm,426664006,AJR
            acute myocardial infarction,57054005,AMI
            acute myocardial ischemia,413444003,AMIs
            anterior ischemia,426434006,AnMIs
            anterior myocardial infarction,54329005,AnMI
            atrial bigeminy,251173003,AB
            atrial fibrillation,164889003,AF
            atrial fibrillation and flutter,195080001,AFAFL
            atrial flutter,164890007,AFL
            atrial hypertrophy,195126007,AH
            atrial pacing pattern,251268003,AP
            atrial tachycardia,713422000,ATach
            atrioventricular junctional rhythm,29320008,AVJR
            av block,233917008,AVB
            blocked premature atrial contraction,251170000,BPAC
            brady tachy syndrome,74615001,BTS
            bradycardia,426627000,Brady
            bundle branch block,6374002,BBB
            cardiac dysrhythmia,698247007,CD
            chronic atrial fibrillation,426749004,CAF
            chronic myocardial ischemia,413844008,CMI
            complete heart block,27885002,CHB
            complete right bundle branch block,713427006,CRBBB
            congenital incomplete atrioventricular heart block,204384007,CIAHB
            coronary heart disease,53741008,CHD
            decreased qt interval,77867006,SQT
            diffuse intraventricular block,82226007,DIB
            early repolarization,428417006,ERe
            fusion beats,13640000,FB
            heart failure,84114007,HF
            heart valve disorder,368009,HVD
            high t-voltage,251259000,HTV
            idioventricular rhythm,49260003,IR
            incomplete left bundle branch block,251120003,ILBBB
            incomplete right bundle branch block,713426002,IRBBB
            indeterminate cardiac axis,251200008,ICA
            inferior ischaemia,425419005,IIs
            inferior ST segment depression,704997005,ISTD
            junctional escape,426995002,JE
            junctional premature complex,251164006,JPC
            junctional tachycardia,426648003,JTach
            lateral ischaemia,425623009,LIs
            left anterior fascicular block,445118002,LAnFB
            left atrial abnormality,253352002,LAA
            left atrial enlargement,67741000119109,LAE
            left atrial hypertrophy,446813000,LAH
            left axis deviation,39732003,LAD
            left bundle branch block,164909002,LBBB
            left posterior fascicular block,445211001,LPFB
            left ventricular hypertrophy,164873001,LVH
            left ventricular strain,370365005,LVS
            low qrs voltages,251146004,LQRSV
            mobitz type i wenckebach atrioventricular block,54016002,MoI
            myocardial infarction,164865005,MI
            myocardial ischemia,164861001,MIs
            nonspecific intraventricular conduction disorder,698252002,NSIVCB
            nonspecific st t abnormality,428750005,NSSTTA
            old myocardial infarction,164867002,OldMI
            pacing rhythm,10370003,PR
            paired ventricular premature complexes,251182009,VPVC
            paroxysmal atrial fibrillation,282825002,PAF
            paroxysmal supraventricular tachycardia,67198005,PSVT
            paroxysmal ventricular tachycardia,425856008,PVT
            premature atrial contraction,284470004,PAC
            premature ventricular contractions,427172004,PVC
            ventricular premature beats,17338001,VPB
            prolonged pr interval,164947007,LPR
            prolonged qt interval,111975006,LQT
            qwave abnormal,164917005,QAb
            r wave abnormal,164921003,RAb
            rapid atrial fibrillation,314208002,RAF
            right atrial abnormality,253339007,RAAb
            right atrial hypertrophy,446358003,RAH
            right axis deviation,47665007,RAD
            right bundle branch block,59118001,RBBB
            right ventricular hypertrophy,89792004,RVH
            s t changes,55930002,STC
            shortened pr interval,49578007,SPRI
            sinoatrial block,65778007,SAB
            sinus arrhythmia,427393009,SA
            sinus bradycardia,426177001,SB
            sinus node dysfunction,60423000,SND
            sinus rhythm,426783006,SNR
            sinus tachycardia,427084000,STach
            st depression,429622005,STD
            st elevation,164931005,STE
            st interval abnormal,164930006,STIAb
            supraventricular bigeminy,251168009,SVB
            supraventricular premature beats,63593006,SVPB
            supraventricular tachycardia,426761007,SVT
            suspect arm ecg leads reversed,251139008,ALR
            t wave abnormal,164934002,TAb
            t wave inversion,59931005,TInv
            transient ischemic attack,266257000,TIA
            u wave abnormal,164937009,UAb
            ventricular bigeminy,11157007,VBig
            ventricular ectopics,164884008,VEB
            ventricular escape beat,75532003,VEsB
            ventricular escape rhythm,81898007,VEsR
            ventricular fibrillation,164896001,VF
            ventricular flutter,111288001,VFL
            ventricular hypertrophy,266249003,VH
            ventricular pacing pattern,251266004,VPP
            ventricular pre excitation,195060002,VPEx
            ventricular tachycardia,164895002,VTach
            ventricular trigeminy,251180001,VTrig
            wandering atrial pacemaker,195101003,WAP
            wolff parkinson white pattern,74390002,WPW
        """

        # rows = open(csv_path).read().strip().split("\n")[1:]
        rows = csv_content.strip().split("\n")[1:]
        #print('Total labels: %d'%(len(rows)))

        dx_map_decode ={}
        dx_map_encode ={}
        for row in rows:
            description, code, abbr= row.strip().split(",")
            dx_map_decode[code] = (abbr,description)
            dx_map_encode[abbr] = code
            
        #YMap
        y_encode={}
        y_decode={}
        
        i=0
        for lbl in self.use_labels:
            y_encode[lbl]=i
            y_decode[i]=lbl
            i=i+1

        return dx_map_decode,dx_map_encode,y_encode, y_decode
        
        
    def generate_cache(self):
        #Make list of all header files
        all_header_files = []         # [(header_file_path, dset)]
        files_of_interest =[]    # [(header_file_path, output_label_array , dset)]

        labelDictTemplate = OrderedDict()
        for lbl in self.use_labels:
            labelDictTemplate[lbl]=0
                    
        for dset in self.use_datasets:
            dataset_dir = self.dataset_paths[dset]                        
            if os.path.isdir(dataset_dir):
                print("Indexing Dataset: ", dset)
                lsdir = os.listdir(dataset_dir)
                #print(lsdir)
                for filename in lsdir:
                    filepath = os.path.join(dataset_dir, filename)
                    if not filename.lower().startswith('.') and filename.lower().endswith('hea') and os.path.isfile(filepath):
                        #all_header_files.append((filepath,dset))
                        with open(filepath, 'r') as f:
                            #print(filepath)
                            for l in f:
                                if l.startswith('#Dx'):
                                    tmp = l.split(': ')[1].split(',')
                                    #print(tmp)
                                    if tmp:
                                        #d = orderedLabelDict()
                                        output_label_array = np.zeros((len(self.use_labels),))                                        
                                        found_interesting_label = False
                                        for c in tmp:
                                            c = c.strip()
                                            c = self.dx_map_decode[c][0]
                                            if c in self.use_labels:                            
                                                #d[c]=1
                                                output_label_array[self.y_encode[c]]=1
                                                found_interesting_label=True

                                        if found_interesting_label:
                                            #output_label_array = label_dict_to_array(d)
                                            files_of_interest.append((filepath,output_label_array , dset))
                                            self.append_recordings(dset,filename,filepath,output_label_array)
                                            
                                            #print(output_label_array)
            
        
    def append_recordings(self,dset,filename,filepath,output_label_array):
        def load_recordings(header_file_path):
            mat_file = header_file_path.replace('.hea', '.mat')
            x = loadmat(mat_file)
            recording = np.asarray(x['val'], dtype=np.float32)
            np.nan_to_num(recording,copy=False) ##Remove NaNs
            return recording 

        def resample(input_signal, Fs_in, Fs_out):
            # input_signal=[[],[],[],...] # 12xL
            n_samples_in = input_signal.shape[1]
            n_samples_out =  round(n_samples_in * Fs_out/Fs_in)

            output_signals=[]
            for i in range(12):
                output_signals.append(signal.resample(input_signal[i], n_samples_out))

            return np.vstack(output_signals)

        rec_name = "%s_%s"%(dset,filename.replace('.hea', ''))
        
        #load, resample
        signals = load_recordings(filepath)
        resampled_signals = resample(signals,self.original_Fs[dset],Fs_out=self.Fs)
        #TODO - normalize & skip recordings with flat values
        #TODO - split resampled_signals
        if rec_name in self.recordings:
            self.recordings[rec_name] = resampled_signals
        else:
            self.recordings.create_dataset(rec_name,data=resampled_signals)
        self.recordings[rec_name].attrs["labels"]=output_label_array

        self.recordings_index.append(rec_name)
        

    def init_cache(self):
        self.cache_file = h5py.File( os.path.join(self.datasets_dir, 'cache.hdf5'), 'w')
        if not "recordings" in self.cache_file:
            self.recordings = self.cache_file.create_group("recordings")        
        else:
            self.recordings = self.cache_file["recordings"]
    def eject_cache(self):
        self.cache_file.close()
                
    def prepare_dataset(self):

        #TODO
        self.init_cache()
        self.generate_cache()
          
       
        
        

        
        pass
        
        
    def setup(self):
        ##
        pass                        
        
    def __getitem__(self,i):
        recfile_name = self.recordings_index[i]        
        return self.recordings[recfile_name][:],self.recordings[recfile_name].attrs["labels"][:]
        
    def __len__(self):
        return len(self.recordings_index)
