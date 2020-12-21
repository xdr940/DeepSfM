from blessings import Terminal
from progressbar import  ProgressBar
import progressbar
import sys
import time


from random import random
def time_formate(sec):
    if sec>3600:
        sec = int(sec)
        h = int(sec / 3600)
        lm = sec % 3600
        m = int(lm / 60)
        s = lm % 60
        ret = '{} h {} m {}s'.format(h,m,s)
    elif sec>60:
        sec = int(sec)
        m = int(sec/60)
        s= sec%60
        ret = '0 h {} m {}s'.format(m,s)
        pass
    else: #0~59
        ret = '0 h 0 m {:.1f}s'.format(sec)




    return ret


def test():
    print('test')
class TermLogger(object):
    def __init__(self, n_epochs, train_size, valid_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.t = Terminal()
        h = self.t.height
        if h == None:
            h = 24  # 终端高度
        space = 13#前面空6行
        space=h-space
        be = 1  # epoch bar position
        bt = 4  # train bar position
        bv = 7  # valid bar position
        #文字框每个三行
        # ----epoch-----
        # err1   err2
        # 1.23   2.34
        # |#####################

        # ----train-----
        # loss1  loss2
        # 1.23   2.34
        # |#####################

        # ----validation-----
        # loss1  loss2
        # 1.23   2.34
        # |23%|#####################
        self.epoch_writer = Writer(self.t, (0, space))
        self.epoch_bar_wirter = Writer(self.t, (0, space+3))

        self.train_writer = Writer(self.t, (0, space+4))#public
        self.train_bar_writer = Writer(self.t, (0, space+7))

        self.valid_writer = Writer(self.t, (0, space+8))#public
        self.valid_bar_writer = Writer(self.t, (0, space+11))

        self.reset_epoch_bar()
        self.reset_train_bar()#152 batches
        self.reset_valid_bar()# 124 batches

    #private
    def reset_epoch_bar(self):
        self.epoch_bar = ProgressBar(maxval=self.n_epochs, fd=self.epoch_bar_wirter).start()
    #public
    def reset_train_bar(self):
        self.train_bar = ProgressBar(maxval=self.train_size, fd=self.train_bar_writer).start()
    #public
    def reset_valid_bar(self):
        self.valid_bar = ProgressBar(maxval=self.valid_size, fd=self.valid_bar_writer).start()
    #public
    def epoch_logger_update(self,epoch,time,names,values):
        headers = ''
        for name in names:
            headers += name + '\t'
        display = '--epochs--[{:d}/{:d}] eduration: {},ETA:{}\n'\
                      .format(epoch,self.n_epochs,time_formate(time),time_formate(time*(self.n_epochs-epoch))) + \
                  headers + \
                  '\n{}'.format(values)

        self.epoch_bar.update(epoch)
        self.epoch_writer.write(display)

    def valid_logger_update(self,batch,time,names,values):
        headers = ''
        for name in names:
            headers += name + '\t'
        display = '----valid--[{:d}/{:d}] batch time {},ETA:{}\n'\
                      .format(batch+1,self.valid_size,time_formate(time),time_formate(time*(self.valid_size-batch))) + \
                  headers + \
                  '\n{}'.format(values)

        self.valid_bar.update(batch)
        self.valid_writer.write(display)

    def train_logger_update(self,batch,time,names,values):
        headers = ''
        for name in names:
            headers += name + '\t'
        display = '--train--[{:d}/{:d}] batch time: {},ETA:{}\n'.\
                      format(batch+1,self.train_size,time_formate(time),time_formate(time*(self.train_size-batch)))\
                  +headers + '\n{}'.format( values)

        self.train_bar.update(batch)
        self.train_writer.write(display)


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        with self.t.location(*self.location):
            sys.stdout.write("\033[K")
            print(string)

    def flush(self):
        return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)



def progressbar_demo1():
    '''
    很像tqdm
    :return:
    '''
    total = 1000
    probar = ProgressBar()
    for i in probar(range(100)):
            time.sleep(0.01)
def progressbar_demo3():
    pass
    total = 100
    pbar =ProgressBar(maxval=total)
    for i in range(total):

        print(i)
        pbar.update(i)
        time.sleep(0.1)
    pbar.finish()

def tqdm_demo():
    '''
    很像tqdm
    :return:
    '''
    total = 1000
    from tqdm import tqdm
    for i in tqdm(range(100)):
        time.sleep(0.01)

def progressbar_demo4():
    import time
    import progressbar
    widgets = [ progressbar.Bar('#'),
                ' [', progressbar.Timer(), '] ',
                progressbar.Percentage(),
                '(', progressbar.ETA(), ') ',]

    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=widgets)
    bar.start()
    for i in range(100):
        bar.update(i)

        time.sleep(0.01)


if __name__ =='__main__':
    print(time_formate(70.5))
    #TermLogger_demo()
    #progressbar_demo4()
    #tqdm_demo()