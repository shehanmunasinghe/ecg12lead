import os

import torch
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm.auto import tqdm

class Trainer():
    def __init__(self,model, loss_fn,optimizer, train_loader, val_loader=None, evaluator=None, model_checkpoint_dir='', tensorboard_log_dir='runs',experiment_name=None ,start_epoch=0, use_tqdm=True):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: ",self.device)

        # Model, loss, optimizer
        self.model = model
        self.model.to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Dataloaders
        self.train_loader=train_loader
        self.val_loader=val_loader

        # Epoch count
        self.epoch = start_epoch

        #Evaluators
        self.evaluator = evaluator

        # Tensorboard SummaryWriter()
        if not os.path.isdir(tensorboard_log_dir):
            raise Exception("Invalid tensorboard_log_dir")
        if not experiment_name:
            date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name='exp_%s'%(date_time)
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(log_dir = os.path.join(tensorboard_log_dir, self.experiment_name))
        print("Writing logs to ",os.path.join(tensorboard_log_dir,experiment_name))

        # Checkpoints directory
        if not os.path.isdir(model_checkpoint_dir):
            raise Exception("Invalid model_checkpoint_dir")
        self.model_checkpoint_dir = model_checkpoint_dir

        #tqdm
        self.use_tqdm = use_tqdm


    def run(self,total_epochs,cpt_interval=5):

        while self.epoch<total_epochs:
            print("Starting Epoch",self.epoch)

            self.train_iter()
            if self.val_loader:
                self.val_iter()

            if cpt_interval:
                if ((self.epoch+1)%cpt_interval)==0:
                    self.save_checkpoint()

            self.epoch+=1
            print("\n")


    def train_iter(self):
        train_loss = 0       
        self.evaluator.reset() 
        
        self.model.train() #Enter train mode

        if self.use_tqdm:
            train_loader = tqdm(self.train_loader)
        else:
            train_loader = self.train_loader

        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(self.device,dtype=torch.float), labels.to(self.device, dtype=torch.float) # Move to device #, dtype=torch.long
            
            self.optimizer.zero_grad() # Clear optimizers            
            output = self.model.forward(inputs) # Forward pass            
            loss = self.loss_fn(output, labels) #            
            loss.backward() # Calculate gradients (backpropogation)            
            self.optimizer.step() # Adjust parameters based on gradients            
            train_loss += loss.item()*inputs.size(0) # Add the loss to the training set's rnning loss

            # Add batch sample into evaluator
            # target = labels.cpu().numpy()
            # pred = output.data.cpu().numpy()
            # pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(labels, output )
                                            
        # Get the average loss for the entire epoch
        train_loss = train_loss/len(self.train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(self.epoch, train_loss))
        self.writer.add_scalar('Loss/train', train_loss, self.epoch)

        # Metrics
        Acc = self.evaluator.get_accuracy()
        # print("Acc: {} \t Acc_class: {}\t mIoU: {}\t fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        self.writer.add_scalar('Acc/train', Acc, self.epoch)

            

    def val_iter(self):
        val_loss = 0

        self.evaluator.reset()
                
        self.model.eval() #Enter eval mode

        if self.use_tqdm:
            val_loader = tqdm(self.val_loader)
        else:
            val_loader = self.val_loader

        with torch.no_grad(): # Tell torch not to calculate gradients
            for inputs, labels in val_loader:
                
                inputs, labels = inputs.to(self.device,dtype=torch.float), labels.to(self.device, dtype=torch.float) # Move to device #, dtype=torch.long    
                            
                output = self.model.forward(inputs) # Forward pass
                
                valloss = self.loss_fn(output, labels) # Calculate Loss                
                val_loss += valloss.item()*inputs.size(0) # Add loss to the validation set's running loss                

                # Add batch sample into evaluator
                # target = labels.cpu().numpy()
                # pred = output.data.cpu().numpy()
                # pred = np.argmax(pred, axis=1)
                self.evaluator.add_batch(labels, output)
 
        # Get the average loss for the entire epoch
        val_loss = val_loss/len(self.val_loader.dataset)
        print('Epoch: {} \tValidation Loss: {:.6f} '.format(self.epoch, val_loss))        
        self.writer.add_scalar('Loss/val', val_loss, self.epoch)

        # Metrics
        Acc = self.evaluator.get_accuracy()
        CM = self.evaluator.get_cm_plot()
        # print("Acc: {} \t Acc_class: {}\t mIoU: {}\t fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        self.writer.add_scalar('Acc/val', Acc, self.epoch)
        self.writer.add_figure('CM/val', CM, self.epoch)

    def save_checkpoint(self):

        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Saving model checkpoint at epoch=%d at  %s"%(self.epoch, date_time))

        cpt_file_name='model_%s_epoch_%d_%s_train.tar'%(self.experiment_name,self.epoch,date_time)
        try:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, 
                os.path.join(self.model_checkpoint_dir,cpt_file_name)
            )
        except: #Google Drive timeout error
            print("Failed to save at specified directory. Saving at present working directory ..") 
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, 
                os.path.join(cpt_file_name)
            )


    def resume(self,cpt_path):

        checkpoint = torch.load(cpt_path) 

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

        print("Loaded model checkpoint from",cpt_path)
        print("Last run epoch=%d"%(checkpoint['epoch']))
        
        self.epoch = checkpoint['epoch'] +1
