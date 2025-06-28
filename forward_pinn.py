import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt


class NN_Burgers(nn.Module): #Fully connected neural network
    def __init__(self, hiddenLayers, hiddenWidth):
        super().__init__()
        layers=[]
        layers.append(nn.Linear(2,hiddenWidth))
        layers.append(nn.Tanh())

        for _ in range(hiddenLayers-1):
            layers.append(nn.Linear(hiddenWidth,hiddenWidth))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hiddenWidth,1))

        self.network=nn.Sequential(*layers)

    def forward(self,t,x):
        input = torch.cat([t,x],dim=1) #dim = 1 to be stacked column per column
        return self.network(input)


def residual(model,t,x):
    u= model(t,x)

    # f= dudt +u*dudx-(0.01/pi)*d2udx2

    dudt= torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True)[0]

    dudx= torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]

    d2udx2= torch.autograd.grad(dudx,x,grad_outputs=torch.ones_like(dudx),create_graph=True)[0]

    return dudt + u*dudx -(0.01/torch.pi)*d2udx2


def boundary_condition_loss(model,t_bc):
    u_left=model(t_bc, torch.full_like(t_bc,-1)) #x=-1
    u_right=model(t_bc, torch.full_like(t_bc,1)) #x= 1

    loss_left= torch.mean(u_left**2) # u(t,-1)=0
    loss_right= torch.mean(u_right**2) # u(t,1)=0

    return loss_left+loss_right

def initial_condition_loss(model,x_ic):
    u_ic=model(torch.full_like(x_ic,0), x_ic) #t=0
    ic_true=-torch.sin(torch.pi*x_ic) #u(0,x)=-sin(pi*x)

    loss_ic= torch.mean((u_ic-ic_true)**2)
    return loss_ic

def physics_loss(model,t_colloc,x_colloc):

    r=residual(model,t_colloc,x_colloc)

    return torch.mean(r**2)

def total_loss(model,t_colloc,x_colloc,t_bc,x_ic,weight_physics=1.0,weight_bc=1.0,weight_ic=1.0):
    p_loss=physics_loss(model,t_colloc,x_colloc)
    bc_loss=boundary_condition_loss(model,t_bc)
    ic_loss=initial_condition_loss(model,x_ic)
    
    return weight_physics*p_loss + weight_ic*ic_loss + weight_bc*bc_loss

def generate_datapoints(N_colloc,N_bc,N_ic,t_min=0,t_max=1,x_min=-1,x_max=1):

    t_colloc=torch.rand(N_colloc,1,requires_grad=True)*(t_max-t_min) + t_min
    x_colloc=torch.rand(N_colloc,1,requires_grad=True)*(x_max-x_min) + x_min

    t_bc=torch.rand(N_bc,1,requires_grad=True)*(t_max-t_min) + t_min

    x_ic=torch.rand(N_ic,1,requires_grad=True)*(x_max-x_min) + x_min

    return t_colloc, x_colloc, t_bc, x_ic

def train_step(model,optimizer, t_colloc, x_colloc, t_bc, x_ic, weight_physics=1.0, weight_bc=1.0, weight_ic= 1.0):
    total_loss_value=total_loss(model,
                                t_colloc,
                                x_colloc,
                                t_bc,
                                x_ic,
                                weight_physics,
                                weight_bc,
                                weight_ic)
    
    #Backpropagation and optimization
    total_loss_value.backward(retain_graph=True) #This is necessary for multiple derivatives -> retain_graph=True
    optimizer.step()
    optimizer.zero_grad()

    return total_loss_value.item()


if __name__ == "__main__":

    N_colloc=5000
    N_bc=100
    N_ic=100

    w_physics=0.8
    w_common=0.2
    w_bc=w_common
    w_ic=w_common

    layers_depth=9
    layers_width=20

    model=NN_Burgers(layers_depth,layers_width)

    #Generate datapoints
    t_colloc, x_colloc, t_bc, x_ic= generate_datapoints(N_colloc,N_bc,N_ic)

    epochs=5000
    optimizer=optim.Adam(model.parameters() ,lr=0.0001)

    train_loss_list=[]

    for e in range(epochs):
        train_loss_value=train_step(model,optimizer,t_colloc,x_colloc,
                              t_bc, x_ic,w_physics, w_bc, w_ic)
        
        train_loss_list.append(train_loss_value)

        if e%100 == 0:
            print(f"Epoch {e}, Total Loss: {train_loss_value:.6f}")

    plt.figure(1)
    plt.plot(range(1,epochs+1),train_loss_list)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")


    # Generate a grid for evaluation
    t_eval = torch.linspace(0, 1, 100).unsqueeze(1)  # 100 points in time , unsqueeze(1) makes the shape of (100,) to (100,1)
    x_eval = torch.linspace(-1, 1, 100).unsqueeze(1)  # 100 points in space, unsqueeze(1) makes the shape of (100,) to (100,1)
    t_grid, x_grid = torch.meshgrid(t_eval.squeeze(), x_eval.squeeze(), indexing="ij")

    # Flatten the grid for model input
    t_flat = t_grid.reshape(-1, 1)
    x_flat = x_grid.reshape(-1, 1)

    # Predict u(t, x) on the grid
    with torch.no_grad():
        u_flat = model(t_flat, x_flat)

    # Reshape the output to match the grid shape
    u_grid = u_flat.reshape(t_grid.shape)

    plt.figure(2)
    plt.contourf(t_grid.numpy(), x_grid.numpy(), u_grid.numpy(), levels=50, cmap="viridis")
    plt.colorbar(label="u(t, x)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Contour Plot of u(t, x)")


    # Slice through t = 0.75
    t_slice = 0.75
    x_slice = x_eval.squeeze()
    t_eval_slice = torch.full_like(x_slice, t_slice).unsqueeze(1)
    x_slice = x_slice.unsqueeze(1)

    # Predict the slice
    with torch.no_grad():
        u_slice = model(t_eval_slice, x_slice)

    # Plot the slice
    plt.figure(3)
    plt.plot(x_slice.numpy(), u_slice.numpy(), label=f"u(x) at t={t_slice}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"Slice of u(x) at t = {t_slice}")
    plt.legend()
    plt.grid()
    plt.show()