from mainFunction import main

from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
from tkinter import *

positionY_directories = 50
positionY_operation = 130
positionY_net = 145
positionY_optimizer = 190
positionX_optimizer = 0
positionY_lossfun = 270
positionX_lossfun = 0

positionY_augs = 50
positionX_augs = 500
positionY_model = 0
positionX_model = 500
positionY_parameters = 0
positionX_parameters = 500


positionY_submit = 200
positionX_submit = 125





def switch(value):
    global switch
  
def sel():
   selection = "You selected the net-structure " + str(var.get())
   #label.config(text = selection)
    
def sel_model():
    selection_model = "You selected the model " + str(var_model.get())
    #label_model.config(text = selection_model)

def sel_optimizer():
    selection_model = "You selected the optimizer " + str(var_optimizer.get())
    #label_model.config(text = selection_model)
    
def sel_lossfun():
    selection_model = "You selected the loss function " + str(var_lossfun.get())
    #label_model.config(text = selection_model)
    
def sel_medmnist():
    D11_var.set(False)
    variables = [D1_var, D2_var, D3_var, D4_var, D5_var, D6_var, D7_var, D8_var, D9_var, D10_var]
    datasets = ["Pathmnist", "Chestmnist", "Dermamnist", "OCTmnist", "Pneumoniamnist", "Retinamnist", "Breastmnist", "Organmnist_axial", "Organmnist_coronal", "Organmnist_sagittal"]
    selection_dataset = "You selected the dataset/s " + ', '.join([str(datasets[x]) for x in range(len(datasets)) if variables[x].get()!=""])
    label_dataset.config(text = selection_dataset)
    global data_name 
    data_name = []
    for var in variables:
        print(var.get())
        if var.get()!= "":
            data_name.append(var.get())
 
def sel_cifar10():
    variables = [D1_var, D2_var, D3_var, D4_var, D5_var, D6_var, D7_var, D8_var, D9_var, D10_var]
    for i in range(len(variables)):
        variables[i].set(False)
    selection_dataset = "You selected the dataset Cifar10"
    label_dataset.config(text = selection_dataset)
      
def submit():
    global operation
    operation = var_switch.get()
    global augmentations
    augmentations = [c1_var.get(), c2_var.get(), c3_var.get(), c4_var.get(), c5_var.get(), c6_var.get(), c7_var.get()]
    global net_input
    net_input = var.get()
    global model
    model = var_model.get()
    global optimizer
    optimizer = var_optimizer.get()
    global lossfun
    lossfun = var_lossfun.get()
    
    if data_name == []:
        print("Please select at least one dataset!")
    elif net_input == "":
        print("Please select Net Structure!")
    elif model == "":
        print("Please select Model!")
    elif optimizer == "":
        print("Please select an optimizer!")
    elif lossfun == "":
        print("Please select a loss function")
    else:
        global num_epoch
        num_epoch = e1_var.get()
        global batch_size
        batch_size = e2_var.get()
        global learning_rate
        learning_rate = e3_var.get()
        global momentum
        momentum = e4_var.get()
        global train_size
        train_size = e5_var.get()*0.01
        global weight_decay
        weight_decay = e6_var.get()
        
        root.destroy()
        for i in range(len(data_name)):
            main(data_name[i],
             data_root_complete.get(),
             output_root_complete.get(),
             num_epoch,
             batch_size,
             learning_rate,
             momentum,
             train_size,
             weight_decay,
             net_input,
             var_switch,
             model,
             optimizer,
             lossfun,
             augmentations,
             True)
        

def browse_data():
    global data_root
    filename = filedialog.askdirectory()
    data_root.set(filename.split("/")[-2] + "/" + filename.split("/")[-1])  
    data_root_complete.set(filename)
        
def browse_output():
    global output_root
    filename = filedialog.askdirectory()
    output_root.set(filename.split("/")[-2] + "/" + filename.split("/")[-1])
    output_root_complete.set(filename)

data_name = []

root = tk.Tk()
root.title('CNN Selector')
root.geometry("900x600")

D1_var = StringVar()
D2_var = StringVar()
D3_var = StringVar()
D4_var = StringVar()
D5_var = StringVar()
D6_var = StringVar()
D7_var = StringVar()
D8_var = StringVar()
D9_var = StringVar()
D10_var = StringVar()
D11_var = StringVar()
 
menubar = tk.Menu(root)
medmnist = tk.Menu(menubar)
medmnist.add_checkbutton(label="PathMnist", variable = D1_var, command=sel_medmnist, onvalue="pathmnist", offvalue="")
medmnist.add_checkbutton(label="ChestMnist", variable = D2_var, command=sel_medmnist, onvalue="chestmnist", offvalue="")
medmnist.add_checkbutton(label="DermaMnist", variable = D3_var, command=sel_medmnist, onvalue="dermamnist", offvalue="")
medmnist.add_checkbutton(label="OCTMnist", variable = D4_var, command=sel_medmnist, onvalue="octmnist", offvalue="")
medmnist.add_checkbutton(label="PneumoniaMnist", variable = D5_var, command=sel_medmnist, onvalue="pneumoniamnist", offvalue="")
medmnist.add_checkbutton(label="RetinaMnist", variable = D6_var, command=sel_medmnist, onvalue="retinamnist", offvalue="")
medmnist.add_checkbutton(label="BreastMnist", variable = D7_var, command=sel_medmnist, onvalue="breastmnist", offvalue="")
medmnist.add_checkbutton(label="OrganMnist axial", variable = D8_var, command=sel_medmnist, onvalue="organmnist_axial", offvalue="")
medmnist.add_checkbutton(label="OrganMnist coronal", variable = D9_var, command=sel_medmnist, onvalue="organmnist_coronal", offvalue="")
medmnist.add_checkbutton(label="OrganMnist sagittal", variable = D10_var, command=sel_medmnist, onvalue="organmnist_sagittal", offvalue="")


cifar10 = tk.Menu(menubar)
cifar10.add_radiobutton(label="full Cifar10", variable = D11_var, value ="cifar10", command=sel_cifar10)

menubar.add_cascade(label="MedMnist", menu=medmnist)
menubar.add_cascade(label="Cifar10", menu=cifar10)

root.config(menu=menubar)

label_dataset = Label(root, text="Please select dataset!",wraplength=700, justify=LEFT, width = 130, height = 4)
label_dataset.place(x=2, y=10)


Label(root, text = "Directories", bd=1, relief="solid", width = 56, height = 2).place(x=2, y=30+positionY_directories)
data_root = StringVar()
data_root_complete = StringVar()
output_root = StringVar()
output_root_complete = StringVar()
data_root_label = Label(master=root,textvariable=data_root, wraplength=250, justify=LEFT).place(x=180, y=70+positionY_directories)
output_root_label = Label(master=root,textvariable=output_root, wraplength=250, justify=LEFT).place(x=180, y=95+positionY_directories)
browse_data_button = Button(text="Browse Dataset Root", command=browse_data).place(x=2, y=65+positionY_directories)
browse_output_button = Button(text="Browse Output Root", command=browse_output).place(x=2, y=90+positionY_directories)




Label(root, text = "Operation", bd=1, relief="solid", width = 56, height = 2).place(x=2, y=60+positionY_operation)
var_switch = BooleanVar()
tk.Label(root, text = "training", bd=1, relief="solid", width = 15, height = 2).place(x=50, y=95+positionY_operation)
switch = tk.Scale(orient = tk.HORIZONTAL,length = 50,to = 1,variable = var_switch, showvalue = False,sliderlength = 25,command = switch)
switch.place(x=170, y=97+positionY_operation)
tk.Label(root, text = "prediction", bd=1, relief="solid", width = 15, height = 2).place(x=235, y=95+positionY_operation)




Label(root, text = "NetStructure", bd=1, relief="solid", width = 56, height = 2).place(x=2, y=130+positionY_net)

var = StringVar()
R1 = Radiobutton(root, text="ResNet-18", variable=var, value="Resnet18",
                  command=sel)
R1.place(x=10, y=160+positionY_net)

R2 = Radiobutton(root, text="ResNet-50", variable=var, value="ResNet50",
                  command=sel)
R2.place(x=10, y=180+positionY_net)

R3 = Radiobutton(root, text="EfficientNet-b0", variable=var, value="EfficientNet-b0",
                  command=sel)
R3.place(x=140, y=160+positionY_net)

R4 = Radiobutton(root, text="EfficientNet-b1", variable=var, value="EfficientNet-b1",
                  command=sel)
R4.place(x=140, y=180+positionY_net)

R5 = Radiobutton(root, text="EfficientNet-b7", variable=var, value="EfficientNet-b7",
                  command=sel)
R5.place(x=140, y=200+positionY_net)

R7 = Radiobutton(root, text="AlexNet", variable=var, value="AlexNet",
                  command=sel)
R7.place(x=310, y=160+positionY_net)

#label = Label(root)
#label.place(x=50, y=145)


Label(root, text = "Optimizer", bd=1, relief="solid", width = 56, height = 2).place(x=2+positionX_optimizer, y=185+positionY_optimizer)

var_optimizer = StringVar()
Op1 = Radiobutton(root, text="SGD", variable=var_optimizer, value="SGD",
                  command=sel_optimizer)
Op1.place(x=10+positionX_optimizer, y=220+positionY_optimizer)

Op2 = Radiobutton(root, text="Adam", variable=var_optimizer, value="Adam",
                  command=sel_optimizer)
Op2.place(x=10+positionX_optimizer, y=240+positionY_optimizer)

Op3 = Radiobutton(root, text="RMSProp", variable=var_optimizer, value="RMSprop",
                  command=sel_optimizer)
Op3.place(x=140+positionX_optimizer, y=220+positionY_optimizer)



Label(root, text = "Loss Function", bd=1, relief="solid", width = 56, height = 2).place(x=2+positionX_lossfun, y=185+positionY_lossfun)

var_lossfun = StringVar()
Loss1 = Radiobutton(root, text="CrossEntropyLoss", variable=var_lossfun, value="crossentropyloss",
                  command=sel_lossfun)
Loss1.place(x=10+positionX_lossfun, y=220+positionY_lossfun)

Loss2 = Radiobutton(root, text="BCE with Logitsloss", variable=var_lossfun, value="bce",
                  command=sel_lossfun)
Loss2.place(x=10+positionX_lossfun, y=240+positionY_lossfun)

Loss3 = Radiobutton(root, text="Max likelihood estimation", variable=var_lossfun, value="MLE",
                  command=sel_lossfun)
Loss3.place(x=180+positionX_lossfun, y=220+positionY_lossfun)

Loss3 = Radiobutton(root, text="Mean Squared Error", variable=var_lossfun, value="MSE",
                  command=sel_lossfun)
Loss3.place(x=180+positionX_lossfun, y=240+positionY_lossfun)




Label(root, text = "Augmentations", bd=1, relief="solid", width = 56, height = 2).place(x=2+positionX_augs, y=30+positionY_augs)
c1_var = StringVar()
c2_var = StringVar()
c3_var = StringVar()
c4_var = StringVar()
c5_var = StringVar()
c6_var = StringVar()
c7_var = StringVar()

c1 = Checkbutton(root, text='centerCrop',variable=c1_var, onvalue="centerCrop", offvalue="")
c1.place(x=10+positionX_augs, y=70+positionY_augs)

c2 = Checkbutton(root, text='colorJitter',variable=c2_var, onvalue="colorJitter", offvalue="")
c2.place(x=10+positionX_augs, y=90+positionY_augs)

c3 = Checkbutton(root, text='gaussianBlur',variable=c3_var, onvalue="gaussianBlur", offvalue="")
c3.place(x=10+positionX_augs, y=110+positionY_augs)

c4 = Checkbutton(root, text='normalize',variable=c4_var, onvalue="normalize", offvalue="")
c4.place(x=140+positionX_augs, y=70+positionY_augs)

c5 = Checkbutton(root, text='randHorFlip',variable=c5_var, onvalue="randomHorizontalFlip", offvalue="")
c5.place(x=140+positionX_augs, y=90+positionY_augs)

c6 = Checkbutton(root, text='randVertFlip',variable=c6_var, onvalue="randomVerticalFlip", offvalue="")
c6.place(x=140+positionX_augs, y=110+positionY_augs)

c7 = Checkbutton(root, text='randRot',variable=c7_var, onvalue="randomRotation", offvalue="")
c7.place(x=310+positionX_augs, y=70+positionY_augs)




Label(root, text = "Model", bd=1, relief="solid", width = 56, height = 2).place(x=2+positionX_model, y=185+positionY_model)

var_model = StringVar()
M1 = Radiobutton(root, text="Pseudolabel", variable=var_model, value="Pseudolabel",
                  command=sel_model)
M1.place(x=10+positionX_model, y=220+positionY_model)

M2 = Radiobutton(root, text="MTSS", variable=var_model, value="MTSS",
                  command=sel_model)
M2.place(x=10+positionX_model, y=240+positionY_model)

M3 = Radiobutton(root, text="NoisyStudent", variable=var_model, value="NoisyStudent",
                  command=sel_model)
M3.place(x=140+positionX_model, y=220+positionY_model)

M4 = Radiobutton(root, text="BaseLine", variable=var_model, value="BaseLine",
                  command=sel_model)
M4.place(x=140+positionX_model, y=240+positionY_model)

#label_model = Label(root)
#label_model.place(x=50, y=330)


Label(root, text = "Parameters", bd=1, relief="solid", width = 56, height = 2).place(x=2+positionX_parameters, y=275+positionY_parameters)

Label(root, text="Epochs").place(x=2+positionX_parameters, y=305+positionY_parameters)
Label(root, text="Batch Size").place(x=2+positionX_parameters, y=325+positionY_parameters)
Label(root, text="Learning Rate").place(x=2+positionX_parameters, y=345+positionY_parameters)
Label(root, text="Momentum").place(x=210+positionX_parameters, y=305+positionY_parameters)
Label(root, text="% of trainset").place(x=210+positionX_parameters, y=325+positionY_parameters)
Label(root, text="Weight Decay").place(x=210+positionX_parameters, y=345+positionY_parameters)

e1_var = IntVar()
e2_var = IntVar()
e3_var = DoubleVar() 
e4_var = DoubleVar() 
e5_var = IntVar()
e6_var = DoubleVar() 

e1_var.set(100)
e2_var.set(8)
e3_var.set(0.001)
e4_var.set(0.9)
e5_var.set(100)
e6_var.set(0)

e1 = Entry(root,textvariable=e1_var).place(x=102+positionX_parameters, y=305+positionY_parameters, width=50)
e2 = Entry(root,textvariable=e2_var).place(x=102+positionX_parameters, y=325+positionY_parameters, width=50)
e3 = Entry(root,textvariable=e3_var).place(x=102+positionX_parameters, y=345+positionY_parameters, width=50)
e4 = Entry(root,textvariable=e4_var).place(x=310+positionX_parameters, y=305+positionY_parameters, width=50)
e5 = Entry(root,textvariable=e5_var).place(x=310+positionX_parameters, y=325+positionY_parameters, width=50)
e6 = Entry(root,textvariable=e6_var).place(x=310+positionX_parameters, y=345+positionY_parameters, width=50)


SubmitButton = Button(root, text="Submit", command=submit)
SubmitButton.place(x=290+positionX_submit, y = 330+positionY_submit)

root.mainloop()

