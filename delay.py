import time ,os,sched

schedule = sched.scheduler(time.time,time.sleep)

def perform_command(cmd,inc):
    os.system(cmd)

def iprint():
    print(time.localtime(time.time()))
def timming_exe(cmd,inc=5):
    schedule.enter(inc,0,perform_command,(cmd,inc))
    iprint()
    schedule.run()


timming_exe('python train.py',inc=20000)
