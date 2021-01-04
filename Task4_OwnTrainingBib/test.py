if dataset_name != 'cifar10':
        # Setting information depending on selected dataset    
        info = INFO[dataset_name]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        val_auc_list = []
        epoch_old = 0
        auc_old = 0
        flag = dataset_name
        dataset_name = dataset_name + "_" + str(train_size)

        dir_path = os.path.join(output_root, '%s_checkpoints' % (dataset_name))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if mode.get() == False: # training
            # creating model and training data
            if net_input == 'Resnet18':
                model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
                train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, 28, augmentations, batch_size, train_size, download)
                print('using ResNet18')
            
            elif net_input == 'Resnet50':
                model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
                train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, 28, augmentations, batch_size, train_size, download)
                print('using ResNet50')
            
            elif net_input == 'EfficientNet-b0':
                model = EfficientNet.from_name('efficientnet-b0',n_channels)
                model._fc= nn.Linear(1280, n_classes)
                model.to(device)
                train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, 224, augmentations, batch_size, train_size, download)
                print('using EfficientNet-b0')
            
            elif net_input == 'EfficientNet-b1':
                model = EfficientNet.from_name('efficientnet-b1',n_channels)
                model._fc= nn.Linear(1280, n_classes)
                model.to(device)
                train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, 224, augmentations, batch_size, train_size, download)
                print('using EfficientNet-b1')
            
            elif net_input == 'EfficientNet-b7':
                model = EfficientNet.from_name('efficientnet-b7',n_channels)
                model._fc= nn.Linear(1280, n_classes)
                model.to(device)
                train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, 224, augmentations, batch_size, train_size, download)
                print('using EfficientNet-b7')
            
            else:
                print("Net not found!")


            # setting up loss function
            if loss_function == "crossentropyloss":
                criterion = nn.CrossEntropyLoss()
            elif loss_function == "bce":
                criterion = nn.BCEWithLogitsLoss()
            elif loss_function == "MSE":
                criterion == nn.MSELoss()
            elif loss_function == "MLE":
                print("actual not supported!")
            else:
                print("failure: using default loss function CE")
                criterion = nn.CrossEntropyLoss()

            # setting up optimizer
            if optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                print('optimizer is SGD')
            elif optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                print('optimizer is Adam')
            elif optimizer == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, centered=False)
                print('optimizer is RMSprop')
            else:
                print("undefined optimizer: taking default SGD")
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            
            if model_input == "Baseline":
                print("creating Baseline-Training")
                auc_list = np.array(val_auc_list)
                index = auc_list.argmax()
                print('epoch %s is the best model' % (index))

                print('==> Testing model...')
                restore_model_path = os.path.join(
                    dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))

                model.load_state_dict(torch.load(restore_model_path)['net'])
                test(model,
                    'train',
                    train_loader,
                    device,
                    dataset_name,
                    task,
                    output_root=output_root)
                test(model, 'val', val_loader, device, dataset_name, task, output_root=output_root)
                test(model,
                    'test',
                    test_loader,
                    device,
                    dataset_name,
                    task,
                    output_root=output_root)
                
                save_best_model_path = os.path.join( 
                    os.path.join(output_root, dataset_name), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
                copyfile(restore_model_path, save_best_model_path)
                shutil.rmtree(dir_path)

            elif model_input == "NoisyStudent":
                print("creating NoisyStudent")

            elif model_input == "MTSS":
                print("training multiple teachers and combine outputs")

            else:
                print("creating pseudo labels for unlabeled dataset")


            # train the model
            for epoch in trange(0, num_epoch):
                train(model, optimizer, criterion, train_loader, device, task)
                epoch_return, auc_return = val(model, val_loader, device, val_auc_list, task, dir_path, epoch, auc_old, epoch_old)
                if auc_return > auc_old:
                    epoch_old = epoch_return
                    auc_old = auc_return

        else: # prediction mode
            print("prediction mode")



    elif data_name == 'cifar10':
        print('cifar10')
    else:
        print('nothing')