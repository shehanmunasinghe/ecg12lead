import matplotlib.pyplot as plt
from ipywidgets import interact

def visualize_dataset(dataset):
    n = len(dataset)

    if not dataset.metadata is None:            
        label_list   = sorted(list(dataset.metadata.keys())[3:])

    def view_recording(i):
        fig, ax = plt.subplots(12,1, figsize=(10,20))
        
        recordings,labels = dataset[i]
        # if not dataset.metadata == None:
        rec_name = "???"
        lbl_str='  '
        if not dataset.metadata is None:
            meta = dataset.metadata.iloc[i]
            dset = meta['_dataset']
            filename = meta['_filename']
            split_no = meta['_split_no']
            rec_name = "%s  %s  %s"%(dset, filename ,split_no)   

            
            for lbl in label_list:
                lbl_str+= "%s:%d  "%(lbl, meta[lbl])

        fig.suptitle(rec_name+lbl_str)
        for j in range(12):
            ax[j].plot(recordings[j])
            # ax[j].set_title("Test")
        plt.show()

    interact(view_recording, i=(0,n-1))