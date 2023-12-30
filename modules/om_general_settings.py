
class OMGeneralSettings():
    def __init__(s):
        s.project_name:str=None
        s.project_dir:str=None
        s.model_cache_folder:str=None
        s.model_file:str=None
        s.save_every_n_epochs:int=None # settings. #1
        s.save_last_n_epochs:int=None # settings. #10
        s.weighted_captions:bool=None # settings. #False
        s.seed:int=None # settings. #42
        s.max_token_length:int=None # settings. #225
        s.xformers:bool=None # settings. #True #True
        s.lowram:bool=None # settings. #False # False
        s.max_data_loader_n_workers:int=None # settings. #0 #8
        s.persistent_data_loader_workers:bool=None # settings. #False #True
        s.v2:bool=None # settings. #False
        s.save_model_as:str=None # settings. #"safetensors"
        s.cache_latents:bool=None # settings. #True 
        return
    
