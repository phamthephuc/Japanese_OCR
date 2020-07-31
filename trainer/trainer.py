import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, strLabelConverter, averager, loadData
from torch.autograd import Variable

import matplotlib.pyplot as plt
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.alphabet = config["alphabet"];
        self.data_loader = data_loader
        self.converter = strLabelConverter(self.alphabet)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        opt = config["data_loader"]["args"]
        batchSize = opt["batch_size"]
        self.image = torch.FloatTensor(batchSize, 3, opt["imgH"], opt["imgH"])
        self.text = torch.IntTensor(batchSize * 5)
        self.length = torch.IntTensor(batchSize)

        if self.is_use_cuda:
            self.image = self.image.cuda()
            self.criterion = self.criterion.cuda()
        self.image = Variable(self.image)
        self.text = Variable(self.text)
        self.length = Variable(self.length)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            batch_size = data.size(0)

            # imshow(make_grid(data))
            # print("batch Size: ", batch_size)
            # data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            loadData(self.image, data)
            output = self.model(self.image)
            t, l = self.converter.encode(target)
            loadData(self.text, t)
            loadData(self.length, l)
            print(target)
            print(l)
            print(t)


            output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
            loss = self.criterion(output, self.text, output_size, self.length) / batch_size
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # self.train_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        # log = self.train_metrics.result()

        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log.update(**{'val_'+k : v for k, v in val_log.items()})

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        return {}

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        loss_avg = averager()
        n_correct = 0
        max_iter = len(self.valid_data_loader)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                batch_size = data.size(0)
                # data, target = data.to(self.device), target.to(self.device)

                text, length = self.converter.encode(target)

                loadData(self.image, data)
                output = self.model(self.image)
                t, l = self.converter.encode(target)
                loadData(self.text, t)
                loadData(self.length, l)

                output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
                loss = self.criterion(output, self.text, output_size, self.length) / batch_size
                loss_avg.add(loss)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.valid_metrics.update('loss', loss_avg.val())

                _, output = output.max(2)
                # preds = preds.squeeze(2)
                output = output.transpose(1, 0).contiguous().view(-1)
                sim_preds = self.converter.decode(output.data, output_size.data, raw=False)
                for pred, tart in zip(sim_preds, target):
                    if pred == tart.lower():
                        n_correct += 1

                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(sim_preds, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        print("output: " , output.data)
        raw_preds = self.converter.decode(output.data, output_size.data, raw=True)[self.test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, target):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        accuracy = n_correct / float(max_iter * batch_size)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

        # self.valid_metrics.update('loss', loss_avg.val())
        # for met in self.metric_ftns:
        #     self.valid_metrics.update(met.__name__, met(sim_preds, target))

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        # return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
