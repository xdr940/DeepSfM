

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def vsd():
    for i in range(1,961):
        print("uav0000239_11136_s/{:07}".format(i))

    pass

def kitti(txt):
    #eigen
    lines =readlines(txt)
    ret_lines=[]
    for line in lines:
        items = line.split(' ')
        ret = items[0]
        if items[2]=='l':
            ret = ret+'/'+'image_02/data'
        else:
            ret = ret+'/'+'image_03/data'
        frame = int(items[1])
        ret= ret + '/{:010d}.png'.format(frame)
        print(ret)
        ret_lines.append(ret)



if __name__ == '__main__':
    #vsd()
    kitti(txt = '/home/roit/datasets/splits/eigen_zhou/test_files.txt')