import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.crnn as module_arch
from parse_config import ConfigParser
import csv
from utils import inf_loop, MetricTracker, strLabelConverter, averager, loadData, loadDataImage, countDifCharacter

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_test', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_obj("loss", module_loss)
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    opt = config["data_loader_test"]["args"]
    batch_size = opt["batch_size"]
    image = torch.FloatTensor(batchSize, 3, opt["imgH"], opt["imgH"])
    text = torch.IntTensor(batchSize * 5)
    length = torch.IntTensor(batchSize)

    if torch.cuda.is_available():
        image = image.cuda()
        criterion = criterion.cuda()
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)


    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    n_correct = 0
    sum_character_error = 0
    max_iter = len(data_loader)
    loss_avg = averager()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):

            batch_size = data.size(0)
            # data, target = data.to(self.device), target.to(self.device)

            text, length = self.converter.encode(target)

            loadData(image, data)
            output = self.model(image)
            t, l = self.converter.encode(target)
            loadData(text, t)
            loadData(length, l)

            output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
            loss = self.criterion(output, text, output_size, length) / batch_size
            loss_avg.add(loss)

            # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            # self.valid_metrics.update('loss', loss_avg.val())

            _, output = output.max(2)
            # preds = preds.squeeze(2)
            output = output.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(output.data, output_size.data, raw=False)
            with open('result.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

                for pred, tart in zip(sim_preds, target):
                    tart = tart.lower
                    if pred == tart:
                        n_correct += 1
                        sum_character_error += countDifCharacter(pred, tart)
                        spamwriter.writerow(tart, pred, "OK")
                    else:
                        spamwriter.writerow(tart, pred, "NG")



            # for met in self.metric_ftns:
            #     self.valid_metrics.update(met.__name__, met(sim_preds, target))
            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        raw_preds = self.converter.decode(output.data, output_size.data, raw=True)[1000]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, target):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        accuracy = n_correct / float(max_iter * batch_size)
        character_error_rate = sum_character_error / float(max_iter * batch_size)
        print('Test loss: %f, accuray: %f, character error rate: %f' % (loss_avg.val(), accuracy, character_error_rate))



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
