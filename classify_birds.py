import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# NOTE: if using GPU:
#   Uncomment lines 46, 106, and use 76 instead of 77
size = 128
checkpoints = "./checkpoints/"

def get_bird_data(augmentation=0):
    transform_train = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=8, padding_mode='edge'), # Take sizexsize crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root='./birds/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='./birds/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    classes = open("./birds/names.txt").read().strip().split("\n")
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
    return {'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name':idx_to_name}

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(net, dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005,
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
    #net.to(device)
    net.train()
    losses = []
    train_accs = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # Load previous training state
    if state:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        losses = state['losses']

    # Fast forward lr schedule through already trained epochs
    for epoch in range(start_epoch):
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

    for epoch in range(start_epoch, epochs):
        sum_loss = 0.0
        start_time = time.time()

        # Update learning rate when scheduled
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

        for i, batch in enumerate(dataloader, 0):
            #inputs, labels = batch[0].to(device), batch[1].to(device)
            inputs, labels = batch[0], batch[1]

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step() # takes a step in gradient direction

            losses.append(loss.item())
            sum_loss += loss.item()

            if i % print_every == print_every-1:    # print every 10 mini-batches
                if verbose:
                    cur_time = time.time()
                    print('[%d, %5d] loss: %.3f elapsed time: %.3f' % (epoch, i + 1, sum_loss / print_every, cur_time - start_time))
                sum_loss = 0.0
        if checkpoint_path:
            train_acc = accuracy_score(net, dataloader)
            train_accs.append(train_acc)
            print("Training accuracy: " + str(train_acc))
            state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
            torch.save(state, checkpoint_path + '-checkpoint-%d.pkl'%(epoch+1))
    return losses, train_accs, net


def predict(net, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    #net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            #images, labels = images.to(device), labels.to(device)
            images, labels = images, labels
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    out.close()


def accuracy_score(model, dataloader) -> float:
    correct = 0.0
    running_accuracy = 0.0
    for batch in dataloader:
        #inputs, labels = batch[0].to(device), batch[1].to(device)
        inputs, labels = batch[0], batch[1]
        y_hat = model(inputs)
        # print("labels: ",labels)
        # print("predicted: ", torch.argmax(y_hat,dim = 1))
        correct = torch.where(torch.argmax(y_hat,dim = 1) == labels,1,0)
        running_accuracy += torch.sum(correct)/len(inputs)
        # print(torch.sum(correct)/len(inputs))
    return running_accuracy.item()/len(dataloader)


if __name__ == '__main__':
    data = get_bird_data()
    dataiter = iter(data['train'])
    images, labels = dataiter.next()
    images = images[:8]

    # If running on GPU:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print("Labels:" + ', '.join('%9s' % data['to_name'][labels[j].item()] for j in range(8)))

    model_names = ["resnet34", "resnet101", "densenet121", "googlenet",
                   "resnext50_32x4d", "efficientnet_b0", "vit_b_16", "convnext_tiny"]

    # Fixed hyperparams
    epochs = 5
    lr = 0.01
    dataloader = data["train"]
    print_every = 10
    for model_name in model_names:
        print("Training: " + model_name + " -------------------------------------------")
        cur_model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
        cur_model.fc = nn.Linear(512, 555) # This will reinitialize the layer as well
        losses, train_accs, model = train(net = cur_model,
                                          dataloader = dataloader,
                                          epochs=epochs,
                                          lr=lr,
                                          print_every = print_every,
                                          checkpoint_path = (checkpoints + model_name))
        np.save(model_name + "train_accs", np.array(train_accs))
        np.save(model_name + "losses", np.array(losses))

    #state = torch.load(checkpoints + 'checkpoint-5.pkl')
    #resnet.load_state_dict(state['net'])
    #predict(resnet, data['test'], checkpoints + "preds.csv")