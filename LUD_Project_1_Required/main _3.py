from model import *
from dataset import MyDataset
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Normalize
from util import setup_seeds
import os
file_name = os.path.splitext(os.path.basename(__file__))[0]



# Use multiple metrics F1 to evaluate




class Trainer:
    def __init__(
            self,
            data_dir: str = "",
            log_dir: str = ".",
            exp_name: str = "log",
            model_name: str = "CNN",
            epochs: int = 50,
            device: str = "cuda:0",
            batch_size: int = 64,
            lr: float = 0.001,
            weight_decay: float = 0.0001,
            random_seed: int = 0,
            optimizer=torch.optim.Adam,
    ):
        setup_seeds(random_seed)
        self.epochs = epochs
        self.device = torch.device(device)

        # Define your model here
        self.model = CNN().to(self.device)

        # Define optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # define transform
        transform = Compose([Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Define dataset and dataloader
        train_dataset = MyDataset(data_dir, transform=transform)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = MyDataset(data_dir, transform=transform, mode='val')
        self.valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MyDataset(data_dir, transform=transform, mode='test')
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


        # Define logger
        self.log_file = log_dir + "/" + exp_name + '_'+file_name+ ".txt"
        f = open(self.log_file, "w")
        info = f"exp name: {exp_name}, model name: {model_name}, epochs: {epochs}, device: {device}, batch size: {batch_size}, lr: {lr}, weight decay: {weight_decay}, random seed: {random_seed}\n"
        f.write(info)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save_checkpoint(self, checkpoint_path: str):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        true_labels = []
        predicted_labels = []

        for x, y in self.valid_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            scores = self.model(x)
            _, predictions = scores.max(1)

            true_labels.extend(y.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

        true_labels = torch.tensor(true_labels)
        predicted_labels = torch.tensor(predicted_labels)

        f1 = f1_score(true_labels, predicted_labels, average='macro')
        return f1

    @torch.no_grad()
    def test(self):
        self.model.eval()
        true_labels = []
        predicted_labels = []

        for x, y in self.test_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            scores = self.model(x)
            _, predictions = scores.max(1)

            true_labels.extend(y.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

        true_labels = torch.tensor(true_labels)
        predicted_labels = torch.tensor(predicted_labels)

        f1 = f1_score(true_labels, predicted_labels, average='macro')
        return f1

    def train(self):
        for epoch in range(self.epochs):
            print("训练轮数{}/{}".format(epoch + 1, self.epochs))

            self.model.train()
            for step, data in enumerate(self.train_dataloader):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                scores = self.model(x)
                loss = self.loss_fn(scores, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 200 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Step [{step}/{len(self.train_dataloader)}], Loss: {loss.item()}")

            # Validate after each epoch
            accuracy = self.validate()
            print(f"Epoch [{epoch + 1}/{self.epochs}], F1: {accuracy}")

        # Log the final results
        final_accuracy = self.test()
        with open(self.log_file, "a") as f:
            f.write(f"F1: {final_accuracy}\n")

        # Save checkpoint
        checkpoint_path = file_name+'_' +"final_checkpoint.pth"
        self.save_checkpoint(checkpoint_path)


def main():
    a = Trainer()
    a.train()
    pass


if __name__ == "__main__":
    main()