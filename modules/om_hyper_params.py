#import keras

class OMHyperParameters():
    def __init__(s):
        s.batch_size:int # use 2^
        s.max_epochs:int=20 # good starting point
        s.num_image_repeats:int=10 # 10 is normal
        s.first_layer_neurons:int # half of features
        s.hidden_layers:int=2 # good starting point
        s.output_nodes:int
        s.learning_rate:float=1e-3 # good starting point
        s.optimizer="Adam" #keras.optimizers.SGD(learning_rate=s.learning_rate) # use any keras optimizer
        s.loss_function="mean_squared_error" # mse default, can use any keras.losses function
        return
    
