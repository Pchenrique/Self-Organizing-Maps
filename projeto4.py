import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from pylab import plot,show,pcolor,colorbar,bone
from minisom import MiniSom

# Getting data from file
dados = np.genfromtxt("colorrectal_2_classes_formatted.txt", delimiter=",")

# Getting classes
classes = dados[:, 142]
classes = classes.astype(int)

# Getting attributes and normalizing
attributes = np.delete(dados,(142), axis=1)
min_max_scaler = MinMaxScaler()
attributes_norm = min_max_scaler.fit_transform(attributes)

for epocas in [100, 200, 300, 1000]:
    
    ### Initialization and training ###
    som = MiniSom(12,12,142,sigma=1.0,learning_rate=1)
    som.random_weights_init(attributes_norm)
    
    som.train_random(attributes_norm,epocas) # training with 100 iterations
    
    bone()
    pcolor(som.distance_map().T) # distance map as background
    colorbar()  
    
    t = classes
    
    # use different colors and markers for each label
    markers = ['', 'o', 's']
    colors = ['', 'r', 'g']
    for cnt,value in enumerate(attributes_norm):
      w = som.winner(value) # getting the winner
      # palce a marker on the winning position for the sample xx
      plot(
        w[0]+.5, 
        w[1]+.5, 
        markers[t[cnt]], 
        markerfacecolor='None', 
        markeredgecolor=colors[t[cnt]], 
        markersize=12, 
        markeredgewidth=2
      )

    show() # show the figure