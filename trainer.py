from datasets import semantickitti
from utils import common
from torch.utils.data import DataLoader

dataset_helper = {
    "semantickitti": semantickitti.SemanticKitti
}

class Trainer:
    def __init__(self, args):
        self.config = common.read_yaml(args.config)
        self._create_dataloader()


    def _create_dataloader(self):

        train_dataset = dataset_helper[self.config["dataset"]](self.config, "train")
        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True, 
            drop_last=False, 
            num_workers=self.config["num_workers"]
        )
        val_dataset = dataset_helper[self.config["dataset"]](self.config, "val")
        self.val_dataloader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False, 
            drop_last=False, 
            num_workers=self.config["num_workers"]
        )



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    trainer = Trainer(args)
    # trainer.run()
