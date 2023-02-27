
import os
import sys
import time
import glob
import scipy
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from Subsets import *
from models import *

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='/home/msiddi01/ImageNet_Vehicles', help='location of the data corpus')
parser.add_argument('--use_model', type=str, default='Conv', help='Sep or Conv')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--valid_size', type=float, default=0, help='validation data size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs') #600
parser.add_argument('--init_channels', type=int, default=24, help='num of init channels') #36
parser.add_argument('--layers', type=int, default=5, help='total number of layers') #20
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EfficientNetB0_ImNet_Vehicles', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')#0
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--class_labels','--list', type=int, nargs='*',default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], help='num classes', required=False)
parser.add_argument('--use_mix', action='store_true', default=False, help='use mix model')
parser.add_argument('--mix_model', nargs="+",default= [0,0,0], type=int)
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  

  #traindir = os.path.join(args.data, 'train')
  #validdir = os.path.join(args.data, 'val')

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_transform= transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
              brightness=0.4,
              contrast=0.4,
              saturation=0.4,
              hue=0.2),
            transforms.ToTensor(),
            normalize,
          ])

  test_transform = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])
  
  if args.dataset == 'cifar10':
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
  
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

  if args.dataset == 'cifar100':
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=train_transform)
  
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

  if args.dataset == 'STL10':
    train_data = torchvision.datasets.STL10(root=args.data_dir, train=True,
                                        download=True, transform=train_transform)
   
    test_data = torchvision.datasets.STL10(root=args.data_dir, train=False,
                                           download=True, transform=test_transform)

  if args.dataset == 'CINIC10':
    train_data = torchvision.datasets.CINIC10(root='./data', train=True,
                                        download=True, transform=train_transform)
   
    test_data = torchvision.datasets.CINIC10(root='./data', train=False,
                                           download=True, transform=test_transform)

  if args.dataset =='flower102':
    train_data = torchvision.datasets.Flowers102(root='./data', split='trainval',
                                        download=True, transform=train_transform)

    test_data = torchvision.datasets.Flowers102(root='./data', split='val',
                                           download=True, transform=test_transform)

  if args.dataset == 'Pets':
    train_data = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval',
                                        download=True, transform=train_transform)
                                        
    test_data = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval',
                                           download=True, transform=test_transform) 

  if args.dataset == 'DTD':
    train_data = torchvision.datasets.DTD(root='./data', split='train',download=True, 
    transform=train_transform)
    test_data = torchvision.datasets.DTD(root='./data', split='val',
                                           download=True, transform=test_transform)

  if args.dataset == 'Food101':
    train_data = torchvision.datasets.Food101(root='/media/abdul/8266fdc5-61f6-47d6-9e82-aa0e56f4b319/Ilyas/classification/food', split='train',download=True, 
    transform=train_transform)
    test_data = torchvision.datasets.Food101(root='/media/abdul/8266fdc5-61f6-47d6-9e82-aa0e56f4b319/Ilyas/classification/food', split='val',download=True, 
    transform=test_transform)

                                        

  # obtain training indices that will be used for validation
  valid_size = args.valid_size
  num_train = len(train_data)
  indices = list(range(num_train))
  np.random.shuffle(indices)
  split = int(np.floor(valid_size * num_train))
  train_idx, valid_idx = indices[split:], indices[:split]
  
  # define samplers for obtaining training and validation batches
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,sampler=train_sampler, num_workers=2)
  valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,sampler=valid_sampler, num_workers=2)

  test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)


# Log sub dataset under consideration.

  class_labels =[]
  if args.dataset == 'cifar10':
    total_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    
  if args.dataset == 'STL10':
    total_classes = ['airplane','bird','car','cat','deer','dogs','horse','monkey','ship','truck']

  if args.dataset == 'Pets':
    total_classes = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound',
     'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau',
      'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 
      'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 
      'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 
      'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 
      'Wheaten Terrier', 'Yorkshire Terrier']
    
  if args.dataset == 'Food101':
     total_classes = ['macarons', 'french_toast', 'lobster_bisque', 'prime_rib', 'pork_chop', 'guacamole', 'baby_back_ribs', 'mussels', 'beef_carpaccio', 'poutine',
                    'hot_and_sour_soup', 'seaweed_salad', 'foie_gras', 'dumplings', 'peking_duck', 'takoyaki', 'bibimbap', 'falafel', 'pulled_pork_sandwich', 'lobster_roll_sandwich',
                    'carrot_cake', 'beet_salad', 'panna_cotta', 'donuts', 'red_velvet_cake', 'grilled_cheese_sandwich', 'cannoli', 'spring_rolls', 'shrimp_and_grits',
                    'clam_chowder','omelette', 'fried_calamari', 'caprese_salad', 'oysters', 'scallops', 'ramen', 'grilled_salmon', 'croque_madame', 'filet_mignon',
                    'hamburger', 'spaghetti_carbonara', 'miso_soup', 'bread_pudding', 'lasagna', 'crab_cakes', 'cheesecake', 'spaghetti_bolognese', 'cup_cakes', 'creme_brulee',
                    'waffles', 'fish_and_chips', 'paella', 'macaroni_and_cheese', 'chocolate_mousse', 'ravioli', 'chicken_curry', 'caesar_salad', 'nachos', 'tiramisu', 'frozen_yogurt',
                    'ice_cream', 'risotto', 'club_sandwich', 'strawberry_shortcake', 'steak', 'churros', 'garlic_bread', 'baklava', 'bruschetta', 'hummus', 'chicken_wings',
                    'greek_salad', 'tuna_tartare', 'chocolate_cake', 'gyoza', 'eggs_benedict', 'deviled_eggs', 'samosa', 'sushi', 'breakfast_burrito', 'ceviche', 'beef_tartare',
                    'apple_pie', '.DS_Store', 'huevos_rancheros', 'beignets', 'pizza', 'edamame', 'french_onion_soup', 'hot_dog', 'tacos', 'chicken_quesadilla', 'pho', 'gnocchi',
                    'pancakes', 'fried_rice', 'cheese_plate', 'onion_rings', 'escargots', 'sashimi', 'pad_thai', 'french_fries']
  
  if args.dataset == 'cifar100':
    total_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
     'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 
     'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
     'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
      'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 
      'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
      'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
       'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
       'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
        'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    
  if args.dataset == 'DTD':
    total_classes = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched',  
              'crystalline', 'dotted', 'fibrous', 'flecked', 'frothy', 'gauzy', 'grid', 'grooved', 'herringbone',   
               'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'patterned',  
               'plaid', 'polka-dotted', 'porous', 'radial', 'ribbed', 'scaled', 'smeared', 'spangled', 'speckled',
               'stippled', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged',
                'homogeneous', 'non-homogeneous']
    
  if args.dataset == 'flower102':
    total_classes = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", 
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
     "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", 
    "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", 
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", 
    "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower", "pelargonium", 
    "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone", 
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", 
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium", 
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", 
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", 
    "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]
    
  for i in range(len(total_classes)):
    class_labels.append(i)
  classes = []
  #print(len(total_classes))
 # exit()
  for i in np.array(class_labels):
    classes.append(total_classes[i])
  #print('Classes under consideration: ', classes)
  logging.info("Classes under consideration: %s", classes)
  ImageNet_CLASSES = len(np.array(class_labels))
  print(ImageNet_CLASSES)
  #exit()

  # Model
  print('==> Building model..')
  # net = VGG('VGG19')
  # net = ResNet18()
  # net = PreActResNet18()
  # net = GoogLeNet()
  # net = DenseNet121()
  # net = ResNeXt29_2x64d()
  # net = MobileNet()
  #net = MobileNetV2()
  # net = DPN92()
  # net = ShuffleNetG2()
  # net = SENet18()
  # net = ShuffleNetV2(1)
  # net = EfficientNetB0()
  #net = RegNetX_200MF()
  #net = net.to(device)

  model = EfficientNetB0(len(total_classes))
  
  #model = MobileNetV2(len(total_classes))
  print(model)
  #exit()
  model = model.cuda()
  #print(model)
  #exit()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    #logging.info('epoch %d', epoch)    

    start_time = time.time()

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    if args.valid_size == 0:
      valid_acc, valid_obj = infer(test_queue, model, criterion)
    else:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)

    logging.info('valid_acc %f', valid_acc)

    end_time = time.time()
    duration = end_time - start_time
    print('Epoch time: %ds.' %duration)

    if valid_acc > best_acc:
      best_acc = valid_acc
      utils.save(model, os.path.join(args.save, 'weights.pt'))

  logging.info('Best Validation Accuracy %f', best_acc)
  utils.load(model, os.path.join(args.save, 'weights.pt'))
  classwisetest(model, classes, test_queue, criterion)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  
  model.train()

  for step, (input, target) in enumerate(train_queue):

    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True) 
    
    #print(input.size(),target.size())
    #exit()


    optimizer.zero_grad()
    logits = model(input)
    #print("---ligita-",logits)
    loss = criterion(logits, target)
    #print("loss",type(loss),loss.size())
    #exit()

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1 = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    #top1.update(prec1.data.item(), n)
    top1.update(prec1, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    with torch.no_grad():
      logits = model(input)
      loss = criterion(logits, target)

    prec1 = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    #top1.update(prec1.data.item(), n)
    top1.update(prec1, n)
    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def classwisetest(model, classes, test_queue, criterion):
    
    num_classes = len(classes)
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    model.eval()
    # iterate over test data
    for data, target in test_queue:
        # move tensors to GPU if CUDA is available        
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        #correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        
        # calculate test accuracy for each object class
        for i in range(len(target)):
            #print(i)
            label = target.data[i]
            #print(label)
            class_correct[label] += correct[i].item()
            class_total[label] += 1
          
           # print(i)
            #print(len(target))

    # average test loss
    test_loss = test_loss/len(test_queue.dataset)
    
    logging.info('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(num_classes):
        if class_total[i] > 0:
            logging.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            logging.info('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    logging.info('\nTest Accuracy (Overall): %2f%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))    



if __name__ == '__main__':
  start_time = time.time()
  main() 
  end_time = time.time()
  duration = end_time - start_time
  logging.info('Total Evaluation Time: %ds', duration)