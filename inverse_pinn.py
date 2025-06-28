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


def residual(model,t, x, lambda_1, lambda_2):
    u= model(t,x)

    dudt= torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True)[0]

    dudx= torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]

    d2udx2= torch.autograd.grad(dudx,x,grad_outputs=torch.ones_like(dudx),create_graph=True)[0]

    return dudt + lambda_1*u*dudx + lambda_2*d2udx2


def data_loss(model,t,x,u_data):
    u_hat=model(t,x)
    
    return torch.mean((u_hat-u_data)**2)


def physics_loss(model, t_colloc, x_colloc, lambda_1, lambda_2):

    r=residual(model,t_colloc,x_colloc, lambda_1, lambda_2)

    return torch.mean(r**2)

def total_loss(model,t_colloc,x_colloc,t_data,x_data,u_data, lambda_1, lambda_2,weight_physics=1.0,weight_data=1e2):
    p_loss=physics_loss(model,t_colloc,x_colloc, lambda_1, lambda_2)
    d_loss=data_loss(model, t_data, x_data, u_data)

    return weight_physics*p_loss + weight_data*d_loss

def generate_datapoints(N_colloc,t_min=0,t_max=1,x_min=-1,x_max=1):

    t_colloc=torch.rand(N_colloc,1,requires_grad=True)*(t_max-t_min) + t_min
    x_colloc=torch.rand(N_colloc,1,requires_grad=True)*(x_max-x_min) + x_min


    return t_colloc, x_colloc

def train_step(model,optimizer, t_colloc, x_colloc, t_data, x_data, u_data, lambda_1, lambda_2, weight_physics=1.0, weight_data=1e2):
    model.train()
    total_loss_value=total_loss(model,
                                t_colloc,
                                x_colloc,
                                t_data,
                                x_data,
                                u_data,
                                lambda_1,
                                lambda_2,
                                weight_physics,
                                weight_data)
    
    #Backpropagation and optimization
    total_loss_value.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss_value.item()

def test_step(model,t_colloc,x_colloc,t_data,x_data,u_data, lambda_1, lambda_2, weight_physics=1.0, weight_data=1e2):
    model.eval()
    total_loss_value=total_loss(model,
                                t_colloc,
                                x_colloc,
                                t_data,
                                x_data,
                                u_data,
                                lambda_1,
                                lambda_2,
                                weight_physics,
                                weight_data)
    
    return total_loss_value.item()


if __name__ == "__main__":
    #Reading data
    data= np.loadtxt("data.txt")
    data= np.array(data)

    # Shuffle the numeric array and the corresponding string array
    nrows = data.shape[0]
    indices = np.arange(nrows)
    np.random.shuffle(indices)
    shuffled_numeric_array = data[indices]

    # Split into training (80%) and test (20%) sets
    split_index = int(0.8 * nrows)

    x_data_train= data[:split_index,0]
    t_data_train= data[:split_index,1]
    u_data_train= data[:split_index,2]

    # Convert numpy arrays to PyTorch tensors
    x_data_train = torch.tensor(x_data_train, dtype=torch.float32).view(-1,1)
    t_data_train = torch.tensor(t_data_train, dtype=torch.float32).view(-1,1)
    u_data_train = torch.tensor(u_data_train, dtype=torch.float32).view(-1,1)


    x_data_test= data[split_index:,0]
    t_data_test= data[split_index:,1]
    u_data_test= data[split_index:,2]

    # Convert numpy arrays to PyTorch tensors
    x_data_test = torch.tensor(x_data_test, dtype=torch.float32).view(-1,1)
    t_data_test = torch.tensor(t_data_test, dtype=torch.float32).view(-1,1)
    u_data_test = torch.tensor(u_data_test, dtype=torch.float32).view(-1,1)


    #Generate datapoints for physics
    N_colloc=5000
    t_colloc, x_colloc= generate_datapoints(N_colloc)

    w_physics=1.0
    w_data=1.0e2

    layers_depth=9
    layers_width=20

    model=NN_Burgers(layers_depth,layers_width)

    
    #Unknown parameters
    lambda_1 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))
    lambda_2 = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))

    epochs=40001
    optimizer=optim.Adam(list(model.parameters()) + [lambda_1] + [lambda_2] ,lr=0.0001)

    train_loss_list=[]
    test_loss_list=[]
    lambda1_list=[]
    lambda2_list=[]


    for e in range(epochs):

        #train
        train_loss_value=train_step(model,optimizer,t_colloc,x_colloc,
                              t_data_train, x_data_train,u_data_train, lambda_1, lambda_2,w_physics, w_data)
        
        train_loss_list.append(train_loss_value)

        #test
        test_loss_value=test_step(model,t_colloc,x_colloc,
                              t_data_train, x_data_train,u_data_train, lambda_1, lambda_2, w_physics, w_data)
        
        test_loss_list.append(test_loss_value)

        lambda1_list.append(lambda_1.detach().cpu().item())  
        lambda2_list.append(lambda_2.detach().cpu().item())  

        if e%1000 == 0:
            print(f"Epoch {e+1}, Train Loss: {train_loss_value:.6f}, Test Loss: {test_loss_value:.6f}, "
                    f"Lambda1: {lambda_1.item():.6f}, Lambda2: {lambda_2.item():.6f}")

    plt.figure(1)
    plt.plot(range(1,epochs+1),train_loss_list,label="Train Loss")
    plt.plot(range(1,epochs+1),test_loss_list,label="Test Loss")
    plt.legend()
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

    # Plot Lambda 1
    exact_lambda1 = np.ones(len(lambda1_list))

    plt.figure(4)
    plt.plot(range(1,epochs+1), lambda1_list, label="Calculated Lambda1")
    plt.plot(range(1,epochs+1), exact_lambda1, label= "Exact value (1.0)")
    plt.xlabel("Epochs")
    plt.ylabel("Lambda 1 Value")
    plt.legend()
    plt.grid(True)

    plt.figure(5)
    exact_lambda2 = np.full(len(lambda2_list), -0.01 / np.pi)
    plt.plot(range(1,epochs+1), lambda2_list, label= "Calculated Lambda 2")
    plt.plot( range(1, epochs+1), exact_lambda2, label= "Exact Value (-0.01/π)")
    plt.xlabel("Epochs")
    plt.ylabel("Lambda 2 Value")
    plt.legend()
    plt.grid(True)

    plt.show()

