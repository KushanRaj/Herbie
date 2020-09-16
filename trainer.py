from datasets import semantickitti,kittiobject,imagenet
from utils import common
from torch.utils.data import DataLoader
from modules import salsanext,YOLO,CSPDarknet53
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

dataset_helper = {
    "semantickitti": semantickitti.SemanticKitti,
    "kittiobject": kittiobject.KittiObject,
    "ImageNet": imagenet.ImageNet
}

model_helper = {
    "salsanext": salsanext.SalsaNext,
    "YOLO": YOLO.Complex_Yolo,
    "Darknet": CSPDarknet53.Detector
}


class Trainer:
    def __init__(self, args):
        self.config = common.read_yaml(args.config)
        self._create_dataloader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._create_model()
        self.writer = SummaryWriter()

    def _create_model(self):
        self.model = model_helper[self.config["model"]](self.config, dataset_helper[self.config["dataset"]], self.device)

    def _create_dataloader(self):
        if "train" in self.config["SPLIT"]:
            train_dataset = dataset_helper[self.config["dataset"]](self.config, "train")
            self.train_dataloader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=True, 
                drop_last=False, 
                num_workers=self.config["num_workers"]
            )
        if "val" in self.config["SPLIT"]:
            val_dataset = dataset_helper[self.config["dataset"]](self.config, "val")
            self.val_dataloader = DataLoader(
                    dataset=val_dataset, 
                    batch_size=self.config["batch_size"], 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=self.config["num_workers"]
                )
    
    def _run(self):
        print ("start training")
        for epoch in tqdm(range(self.config["epochs"])):
            train_log = self.model.train(self.train_dataloader, self.writer)
            print(f"train metrics: loss - {train_log['loss']}  accuracy - {train_log['acc']}")
            if epoch % self.config["valid_every"]:
                
                self.model.save(f'/weights/{self.config["model"]}/{epoch}.pth')
                if "val" in self.config["SPLIT"]:
                    valid_log = self.model.valid(self.val_dataloader, self.writer)
                    print(f"valid metrics: loss - {valid_log['loss']}  accuracy - {valid_log['acc']}")
        
    
    def close_writer(self):
        self.writer.close()
    def load_model(self,path):
        self.model.load_model(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer._run()
