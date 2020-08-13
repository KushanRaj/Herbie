from datasets import semantickitti
from utils import common
from torch.utils.data import DataLoader
from modules import salsanext

dataset_helper = {
    "semantickitti": semantickitti.SemanticKitti
}

model_helper = {
    "salsanext": salsanext.SalsaNext
}


class Trainer:
    def __init__(self, args):
        self.config = common.read_yaml(args.config)
        self._create_dataloader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._create_model()

    def _create_model(self):
        self.model = model_helper[self.config["model"]](self.config, dataset_helper[self.config["dataset"]], self.device)

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
    
    def _run(self):
        for epoch in range(self.config["epochs"]):
            train_log = self.model.train(self.train_dataloader, self.writer)
            if epoch % self.config["valid_every"]:
                valid_log = self.model.valid(self.val_dataloader, self.writer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    trainer = Trainer(args)
    # trainer.run()
