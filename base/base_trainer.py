import os
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.loss = loss
        self.metrics = metrics

        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']

        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        self.device = torch.device('cuda:' + str(config['gpu']) if self.with_cuda else 'cpu')
        self.model = self.model.to(self.device)

        self.train_logger = train_logger
        self.optimizer = optimizer

        # monitor model performance with given metric and save best model to model_best checkpoint
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1

        # setup checkpoint, visualization path using timestamp
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time)
        writer_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, config['visualization']['tensorboardX'])

        # Save configuration into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            try:
                best = (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best)\
                    or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best)
                if best:
                    self.monitor_best = log[self.monitor]
            except KeyError:
                if epoch == 1:
                    msg = "Warning: Can\'t recognize metric named '{}' for performance monitoring. ".format(self.monitor)\
                        + "model_best checkpoint will be overridden every steps."
                    self.logger.warning(msg)
                best = True
 
            self._save_checkpoint(epoch, save_best=best)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth.tar'.format(epoch))
        torch.save(state, filename)
        if save_best:
            os.replace(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.device)
        self.train_logger = checkpoint['logger']
        self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
