from mainFunction import main

from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
from tkinter import *


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

def printValues(choices):
    for name, var in choices.items():
        print("%s: %s" % (name, var.get()))
        
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
 
menubutton = tk.Menubutton(text="Select Dataset",indicatoron=True, borderwidth=1, relief="raised")
menu = tk.Menu(menubutton, tearoff=False)
menubutton.configure(menu=menu)
menubutton.grid(row=0 ,column = 4, columnspan=4, padx='5', pady='5',sticky='ew')

labels = ["PathMnist", "ChestMnist", "Dermamnist", "OCTMnist", "PneumoniaMnist", "RetinaMnist", "BreastMnist", "OrganMnist axial", "OrganMnist coronal", "OrganMnist sagittal"]
onvalues = ["pathmnist", "chestmnist", "dermamnist","octmnist", "pneumoniamnist", "retinamnist", "breastmnist", "organmnist_axial", "organmnist_coronal", "organmnist_sagittal"]
variables = [D1_var, D2_var, D3_var, D4_var, D5_var, D6_var, D7_var, D8_var, D9_var, D10_var, D11_var]
for choice in range(len(labels)):
    menu.add_checkbutton(label=labels[choice], variable=variables[choice], 
                         onvalue=onvalues[choice], offvalue="", 
                         command=sel_medmnist)


label_dataset = Label(root, text="Please select dataset!",height = 4, wraplength=250)
label_dataset.grid(row=0, padx='5', pady='5',sticky='ew')


Label(root, text = "Directories", bd=1, relief="solid", width = 56, height = 2).grid(row=1 ,column = 0, padx='0', pady='0',sticky='ew',columnspan=3)
data_root = StringVar()
data_root_complete = StringVar()
output_root = StringVar()
output_root_complete = StringVar()
data_root_label = Label(master=root,textvariable=data_root, wraplength=250, justify=LEFT).grid(row=2 ,column = 1, padx='5', pady='5',sticky='ew')
output_root_label = Label(master=root,textvariable=output_root, wraplength=250, justify=LEFT).grid(row=3 ,column = 1, padx='5', pady='5',sticky='ew')
browse_data_button = Button(text="Browse Dataset Root", command=browse_data).grid(row=2 ,column = 0, padx='5', pady='5',sticky='ew')
browse_output_button = Button(text="Browse Output Root", command=browse_output).grid(row=3 ,column = 0, padx='5', pady='5',sticky='ew')

#Setting Defaults
data_root.set("/DataSets/medmnist")
data_root_complete.set("/Users/gerrit/Documents/Master-Thesis/DataSets/medmnist")
output_root.set("/Master-Thesis/TrainedNets/")
output_root_complete.set("/Users/gerrit/Documents/Master-Thesis/TrainedNets/")


Label(root, text = "Operation", bd=1, relief="solid", width = 56, height = 2).grid(row=4 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')
var_switch = BooleanVar()
var_switch.set(False)
tk.Label(root, text = "training", bd=1, relief="solid", width = 15, height = 2).grid(row=5 ,column = 0, padx='5', pady='5',sticky='ew')
switch = tk.Scale(orient = tk.HORIZONTAL,length = 50,to = 1, variable = var_switch, showvalue = False,sliderlength = 25,command = switch)
switch.grid(row=5 ,column = 1, padx='5', pady='5',sticky='ew')
tk.Label(root, text = "prediction", bd=1, relief="solid", width = 15, height = 2).grid(row=5 ,column = 2, padx='5', pady='5',sticky='ew')




Label(root, text = "NetStructure", bd=1, relief="solid", width = 56, height = 2).grid(row=7 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')

var = StringVar()
R1 = Radiobutton(root, text="ResNet-18", variable=var, value="Resnet18",
                  command=sel)
R1.grid(row=8 ,column = 0, padx='5', pady='5',sticky='ew')

R2 = Radiobutton(root, text="ResNet-50", variable=var, value="ResNet50",
                  command=sel)
R2.grid(row=9 ,column = 0, padx='5', pady='5',sticky='ew')

R3 = Radiobutton(root, text="EfficientNet-b0", variable=var, value="EfficientNet-b0",
                  command=sel)
R3.grid(row=8 ,column = 1, padx='5', pady='5',sticky='ew')

R4 = Radiobutton(root, text="EfficientNet-b1", variable=var, value="EfficientNet-b1",
                  command=sel)
R4.grid(row=9 ,column = 1, padx='5', pady='5',sticky='ew')

R5 = Radiobutton(root, text="EfficientNet-b7", variable=var, value="EfficientNet-b7",
                  command=sel)
R5.grid(row=10 ,column = 1, padx='5', pady='5',sticky='ew')

R7 = Radiobutton(root, text="AlexNet", variable=var, value="AlexNet",
                  command=sel)
R7.grid(row=8 ,column = 2, padx='5', pady='5',sticky='ew')

#label = Label(root)
#label.place(x=50, y=145)


Label(root, text = "Optimizer", bd=1, relief="solid", width = 56, height = 2).grid(row=11 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')

var_optimizer = StringVar()
Op1 = Radiobutton(root, text="SGD", variable=var_optimizer, value="SGD",
                  command=sel_optimizer)
Op1.grid(row=12 ,column = 0, padx='5', pady='5',sticky='ew')

Op2 = Radiobutton(root, text="Adam", variable=var_optimizer, value="Adam",
                  command=sel_optimizer)
Op2.grid(row=12 ,column = 1, padx='5', pady='5',sticky='ew')

Op3 = Radiobutton(root, text="RMSProp", variable=var_optimizer, value="RMSprop",
                  command=sel_optimizer)
Op3.grid(row=12 ,column = 2, padx='5', pady='5',sticky='ew')



Label(root, text = "Loss Function", bd=1, relief="solid", width = 56, height = 2).grid(row=13 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')

var_lossfun = StringVar()
Loss1 = Radiobutton(root, text="CrossEntropyLoss", variable=var_lossfun, value="crossentropyloss",
                  command=sel_lossfun)
Loss1.grid(row=14 ,column = 0, padx='5', pady='5',sticky='ew')

Loss2 = Radiobutton(root, text="BCE with Logitsloss", variable=var_lossfun, value="bce",
                  command=sel_lossfun)
Loss2.grid(row=15 ,column = 0, padx='5', pady='5',sticky='ew')

Loss3 = Radiobutton(root, text="Max likelihood estimation", variable=var_lossfun, value="MLE",
                  command=sel_lossfun)
Loss3.grid(row=14 ,column = 1, padx='5', pady='5',sticky='ew')

Loss3 = Radiobutton(root, text="Mean Squared Error", variable=var_lossfun, value="MSE",
                  command=sel_lossfun)
Loss3.grid(row=15 ,column = 1, padx='5', pady='5',sticky='ew')




Label(root, text = "Augmentations", bd=1, relief="solid", width = 56, height = 2).grid(row=1 ,column = 4,columnspan=4, padx='0', pady='0',sticky='ew')
c1_var = StringVar()
c2_var = StringVar()
c3_var = StringVar()
c4_var = StringVar()
c5_var = StringVar()
c6_var = StringVar()
c7_var = StringVar()

c1 = Checkbutton(root, text='centerCrop',variable=c1_var, onvalue="centerCrop", offvalue="")
c1.grid(row=2 ,column = 4, padx='5', pady='5',sticky='ew')

c2 = Checkbutton(root, text='colorJitter',variable=c2_var, onvalue="colorJitter", offvalue="")
c2.grid(row=3 ,column = 4, padx='5', pady='5',sticky='ew')

c3 = Checkbutton(root, text='gaussianBlur',variable=c3_var, onvalue="gaussianBlur", offvalue="")
c3.grid(row=4 ,column = 4, padx='5', pady='5',sticky='ew')

c4 = Checkbutton(root, text='normalize',variable=c4_var, onvalue="normalize", offvalue="")
c4.grid(row=2 ,column = 5, padx='5', pady='5',sticky='ew')

c5 = Checkbutton(root, text='randHorFlip',variable=c5_var, onvalue="randomHorizontalFlip", offvalue="")
c5.grid(row=3 ,column = 5, padx='5', pady='5',sticky='ew')

c6 = Checkbutton(root, text='randVertFlip',variable=c6_var, onvalue="randomVerticalFlip", offvalue="")
c6.grid(row=4 ,column = 5, padx='5', pady='5',sticky='ew')

c7 = Checkbutton(root, text='randRot',variable=c7_var, onvalue="randomRotation", offvalue="")
c7.grid(row=2 ,column = 6, padx='5', pady='5',sticky='ew')




Label(root, text = "Model", bd=1, relief="solid", width = 56, height = 2).grid(row=4 ,column =4, columnspan=4, padx='0', pady='0',sticky='ew')

var_model = StringVar()
M1 = Radiobutton(root, text="Pseudolabel", variable=var_model, value="Pseudolabel",
                  command=sel_model)
M1.grid(row=5 ,column = 4, padx='5', pady='5',sticky='ew')

M2 = Radiobutton(root, text="MTSS", variable=var_model, value="MTSS",
                  command=sel_model)
M2.grid(row=6 ,column = 4, padx='5', pady='5',sticky='ew')

M3 = Radiobutton(root, text="NoisyStudent", variable=var_model, value="NoisyStudent",
                  command=sel_model)
M3.grid(row=5 ,column = 5, padx='5', pady='5',sticky='ew')

M4 = Radiobutton(root, text="BaseLine", variable=var_model, value="BaseLine",
                  command=sel_model)
M4.grid(row=6 ,column = 5, padx='5', pady='5',sticky='ew')

#label_model = Label(root)
#label_model.place(x=50, y=330)


Label(root, text = "Parameters", bd=1, relief="solid", width = 56, height = 2).grid(row=7 ,column = 4, columnspan=4, padx='0', pady='0',sticky='ew')

Label(root, text="Epochs").grid(row=8 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Batch Size").grid(row=9 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Learning Rate").grid(row=10 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Momentum").grid(row=8 ,column = 6, padx='5', pady='5',sticky='ew')
Label(root, text="% of trainset").grid(row=9 ,column = 6, padx='5', pady='5',sticky='ew')
Label(root, text="Weight Decay").grid(row=10 ,column = 6, padx='5', pady='5',sticky='ew')

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

e1 = Entry(root,textvariable=e1_var).grid(row=8 ,column = 5, padx='5', pady='5',sticky='ew')
e2 = Entry(root,textvariable=e2_var).grid(row=9 ,column = 5, padx='5', pady='5',sticky='ew')
e3 = Entry(root,textvariable=e3_var).grid(row=10 ,column = 5, padx='5', pady='5',sticky='ew')
e4 = Entry(root,textvariable=e4_var).grid(row=8 ,column = 7, padx='5', pady='5',sticky='ew')
e5 = Entry(root,textvariable=e5_var).grid(row=9 ,column = 7, padx='5', pady='5',sticky='ew')
e6 = Entry(root,textvariable=e6_var).grid(row=10 ,column = 7, padx='5', pady='5',sticky='ew')


SubmitButton = Button(root, text="Submit", command=submit)
SubmitButton.grid(row=12 ,column = 5, columnspan=2, padx='5', pady='5',sticky='ew')

root.mainloop()

