from torch.nn import Module
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch.nn as nn
from .bert import BERT
from .bert_lm import BERTLM
from .optim_schedule import ScheduledOptim
import tqdm

class Trainer(Module):

    def __init__(self, dataloader: DataLoader, model, loss_fn, optimizer, epoch, lr= 0.01, Tester= False, temp_file= './temp/'):

        super(Trainer, self).__init__()
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.Tester = Tester
        self.temp_file = temp_file
        self.epoch = epoch

    def forward(self, other_component= None):

        if self.Tester == False:
            train_writer = SummaryWriter(self.temp_file)
            self.model.train()
            self.model.zero_grad()
            for i in torch.arange(0, self.epoch):
                for j, data in enumerate(self.dataloader):
                    y_pre = self.model(data['text'], *other_component)
                    loss = self.loss_fn(y_pre, data['label'])
                    loss.backward()
                    self.optimizer.step()
                    train_writer.add_scalar('train_loss', loss.item(), j + 1000 * i)
        else:
            self.model.test()


class BERTTrainer:


    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):


        # cuda标志
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # 每次epoch会保存一次模型
        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # 如果有多块GPU将会使用分布式训练
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "test"

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            next_loss = self.criterion(next_sent_output, data["is_next"])

            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            loss = next_loss + mask_loss

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):

        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path