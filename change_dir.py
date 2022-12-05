import os 
import shutil
import traceback
#用来整理生成图片的
def move_file(src_path, dst_path, file):
    # cmd = 'chmod -R +x ' + src_path
    # os.popen(cmd)
    f_src = os.path.join(src_path, file)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    f_dst = os.path.join(dst_path, file)
    shutil.move(f_src, f_dst)

dir = './cifar10_original_generation'
image_names = os.listdir(dir)
for image_name in image_names:
    temp = image_name.split('.')
    # print('temp=',temp)
    if(len(temp) == 1):
        continue
    num = (int)(temp[0])
    num_divide = num // 1000
    # print("num_divide=",num_divide)
    num = num_divide * 1000
    if(num == 0):
        num = '0000'
    # print('./cifar10_original_generation_test/'+(5-len(str(num)))*'0'+str(num)+'.png')
    # os.rename('./testdir/'+image_name,'./testdir/'+(5-len(str(num)))*'0'+str(num)+'.png')
    # print(image_name)
    # print('./cifar10_original_generation_test/'+(5-len(str(num)))*'0'+str(num))
    move_file('./cifar10_original_generation/','./cifar10_original_generation/'+\
            (5-len(str(num)))*'0'+str(num),image_name)
    
    # dir_name = dir + '/' + fill * '0' + str(num_) 

    # move_file('./cifar10_original_generation/00000',dir,image_name)
