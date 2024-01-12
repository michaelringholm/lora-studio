import streamlit as st
import sys
import csv
import os
import random as rnd
import modules.om_lora_trainer as omlt
import json
import traceback as trc
import modules.captions.make_captions as mkcap
import modules.om_logging as oml
import modules.om_observer as omo
import modules.om_hyper_params as omhp
import modules.om_general_settings as omgs

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'mymodules'))

class App():
    def __init__(s):
        oml.debug("__init__ called!")

    def init_streamlit(s):
        st.set_page_config(page_title='LoRA Studio', page_icon='🌇') 
        st.header('LoRA Studio 🌇')
        sidebar=st.sidebar
        sidebar.header('Advanced')
        #selectedAddressOption=sidebar.selectbox("Select city", options=list(options.keys()), index=st.session_state['SELECTED_HOUSE_INDEX'])
        #with st.sidebar

    def useful_icons(s):
        # AI: Print a set of characters for m2 and heating, power etc. all related to housing information for easy use in a streamlit markdown section.
        # 🔥: Has heating
        # ❄️: No heating 
        ## Power
        # ⚡: Electricity 
        # 💡: Solar power
        # 🔋: Battery
        ## Water 
        # 💧: Running water
        # 🚰: Well water
        # Power: 🔌
        # Kitchen: 🍳
        # Bathroom: 🚿
        # Carport: 🅿️
        # 🏠 - House
        # 🌇 - View
        # Bathroom: 🛁
        # Bedroom: 🛏️
        # Dice: 🎲    
        # Calendar: 🗓️
        # Money: 💰
        # Robots and AI
        # 🤖: Robot Face
        # 🧑‍💻: Person Coding
        # 🤯: Mind Blown
        # 🤖🧠: Robot Brain
        # 🤖💬: Robot Speaking Head
        # 🤖🤖: Two Robots
        # Nature
        # 🌳: Tree
        # 🌺: Flower
        # 🌊: Wave
        # ☀️: Sun
        # 🌙: Moon
        # 🌈: Rainbow
        # Weather
        # ☔: Umbrella (Rain)
        # ❄️: Snowflake
        # 🌪️: Tornado
        # Transportation
        # 🚗: Car
        # 🚲: Bicycle
        # 🚀: Rocket
        # ✈️: Airplane
        # 🚢: Ship
        # Food and Drink
        # 🍎: Apple
        # 🍕: Pizza
        # 🍔: Hamburger
        # 🍦: Ice Cream
        # 🍹: Tropical Drink
        # ☕: Coffee
        # Faces and Emotions
        # 😊: Smiling Face
        # 😢: Crying Face
        # 😎: Cool Face
        # 😍: Heart Eyes
        # Symbols
        # 💻: Laptop
        # 📱: Mobile Phone
        # 💼: Briefcase
        # 📚: Book        
        return

    def draw_footer(s):
        footer = st.container()
        with open('css/custom.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        footer_html = """
        <footer>
        Made by Opus Magus with ❤️
        </footer>
        """
        st.markdown(footer_html, unsafe_allow_html=True)
        return

    def build_hyper_parameters(s):
        hyper_parameters=omhp.OMHyperParameters()
        hyper_parameters.batch_size=64
        hyper_parameters.num_epochs=30
        return hyper_parameters
    
    def update_training_progress(s,epoch,step,current_loss,avg_loss,global_step):
        total_steps=int(s.total_steps)
        progress=(global_step)/total_steps # s.hyper_parameters.num_epochs*20
        if(progress>1):
            progress=1
            oml.warn(f"update_training_progress.progress>1, global_step={global_step},total_steps={total_steps}")
        s.training_progress_bar.progress(progress)
        s.progress_bar_text.markdown(f"👟step {global_step}/{total_steps}, 📈current_loss={current_loss}, 📈avg_loss={avg_loss}") # TODO use steps s.hyper_parameters.num_epochs
        s.current_loss_data.append({"step":global_step, "current_loss":current_loss})    
        s.avg_loss_data.append({"step":global_step, "avg_loss":avg_loss})
        s.current_loss_line_chart_placeholder.line_chart(s.current_loss_data,x="step",y="current_loss")
        s.avg_loss_line_chart_placeholder.line_chart(s.avg_loss_data,x="step",y="avg_loss")
        return
    
    def update_cache_latents_progress(s,image_batch,num_batches,next_image_batch):
        #oml.debug(f"image_batch={image_batch} with type={type(image_batch)} and len(image_batch)={len(image_batch)},num_batches={num_batches},next_image_batch={next_image_batch}")
        progress=(next_image_batch/num_batches) # s.hyper_parameters.num_epochs*20
        s.latent_cache_progress_bar.progress(progress)
        return
    
    def update_training_start_meta_data(s,pre_steps_per_epoch,steps_per_epoch,total_steps,estimated_epochs):
        s.total_steps=s.total_steps_placeholder.text_input("Total Steps 👟",value=total_steps)
        s.estimated_epochs_placeholder.text_input("Estimated Epocs 📆",value=estimated_epochs)
        return    
    
    def update_cuda_info(s,gpu,cuda_installed):
        s.env_widget.markdown(f"🖥️ {gpu.name}")
        cuda_installed_symbol='✅' if cuda_installed else '🚫'
        s.env_widget.markdown(f"📦CUDA {cuda_installed_symbol}")
        return
    
    def update_training_result(s,loss,val_loss):
        #s.training_result_widget.text(f"Loss={loss}, Val_Loss={val_loss}")
        s.training_result_text.text(f"Loss={loss}, Val_Loss={val_loss}")
        return

    def display_data(s,train_data_features,train_data_target,eval_data_features,eval_data_target):
        cols=s.body.columns(2)
        col1=cols[0]
        col2=cols[1]
        col1.expander("Training Features").table(train_data_features)
        col2.expander("Training Target Values").table(train_data_target)
        col1.expander("Evaluation Features").table(eval_data_features)
        col2.expander("Evaluation Target Values").table(eval_data_target)
        return

    def draw_training_progress_widget(s):
        widget=s.body.container()
        widget.subheader("Training progress")
        s.training_status_ph=st.empty()
        widget.markdown("**Latent Cache**")
        s.latent_cache_progress_bar=widget.progress(value=0)
        widget.markdown("**👟Training Steps**")
        s.training_progress_bar=widget.progress(value=0)
        s.progress_bar_text=widget.empty()
        s.current_loss_data=[{"step":None, "curret_loss":None}]
        s.avg_loss_data=[{"step":None, "avg_loss":None}]
        s.current_loss_line_chart_placeholder=widget.empty() #widget.line_chart(s.loss_data,x="epoch",y="loss")
        s.avg_loss_line_chart_placeholder=widget.empty() #widget.line_chart(s.loss_data,x="epoch",y="loss")
        return
    
    def draw_training_result_widget(s):
        s.training_result_widget=s.body.container(border=True)
        s.training_result_widget.subheader("Training result")
        s.training_result_text=s.training_result_widget.text("Awaiting training job")
        return
    
    def draw_input_data_widget(s):
        widget=s.body.expander("Input Data")
        cols=widget.columns(2)
        col1=cols[0]
        col2=cols[1]
        col1.text_input(label="Filename",value="data/stock_prices.csv")
        col1.slider(label="Limit dataset",min_value=1,max_value=100000000,value=1000)
        #col1.slider(label="Learning Rate",min_value=1e-6,max_value=1e-2,value=1e-3)
        col2.text_input(label="Target column",placeholder="Column to predict")
        col2.text_input(label="Date formats",placeholder="%d-%y-%m")
        #col2.slider(label="Output Nodes",min_value=1,max_value=10,value=1)        
        return
    
    def draw_hyper_parameter_widget(s):
        widget=s.body.expander("Hyper Parameters 🎛️")
        cols=widget.columns(2)
        col1=cols[0]
        col2=cols[1]
        s.hyper_parameters.max_epochs=col1.slider(label="Epochs",min_value=1,max_value=40,value=1)
        s.hyper_parameters.batch_size=col1.slider(label="Batch Size",min_value=1,max_value=128,value=2)
        display_factor=10000
        s.hyper_parameters.learning_rate=(col1.slider(label="🏎️Learning Rate",min_value=1,max_value=100,step=10,value=int(4e-3*display_factor))/display_factor)
        s.hyper_parameters.first_layer_neurons=col2.slider(label="First Layer Neurons",min_value=1,max_value=1000,value=30)
        s.hyper_parameters.hidden_layers=col2.slider(label="Hidden Layers",min_value=1,max_value=10,value=2)
        s.hyper_parameters.output_nodes=col2.slider(label="Output Nodes",min_value=1,max_value=10,value=1)   
        s.hyper_parameters.num_image_repeats=col1.slider(label="Image Repeats",value=5,min_value=1,max_value=20)          
        s.hyper_parameters.clip_skip=col2.slider(label="Clip Skip",value=2,min_value=1,max_value=4)        
        s.hyper_parameters.optimizer_name=col1.selectbox("Optimizer", ["Adam","AdamW8bit","Adamax","AdamW","Adagrad"],index=0)
        s.hyper_parameters.lr_scheduler=col2.selectbox("LR Scheduler", ["cosine_with_restarts"],index=0) # "cosine_with_restarts"
        s.hyper_parameters.unet_lr=(col1.slider(label="UNET LR",min_value=1,max_value=100,value=int(5e-4*display_factor))/display_factor) # 0.0005
        s.hyper_parameters.text_encoder_lr=(col2.slider(label="Text Encoder LR",min_value=1,max_value=1000,value=int(1e-4*display_factor))/display_factor) # 0.0001
        s.hyper_parameters.network_dim=col1.slider(label="Network Dim",min_value=2,max_value=32,value=16,step=2) # 16
        s.hyper_parameters.network_alpha=col2.slider(label="Network Alpha",min_value=2,max_value=32,value=8,step=2) # 8
        s.hyper_parameters.network_module=col1.selectbox("Network Module", ["networks.lora"],index=0) # "networks.lora"
        s.hyper_parameters.lr_warmup_steps=col2.slider(label="LR Warmup Steps",min_value=50,max_value=75,value=65) # 65 # None for Adafactor otherwise around 65        
        s.hyper_parameters.lr_scheduler_num_cycles=col1.slider(label="LR Scheduler Cycles",min_value=1,max_value=10,value=3) # 3    
        return
    
    def draw_training_meta_data(s):
        widget=s.body.expander("Training Meta Data")
        cols=widget.columns(2)
        col1=cols[0]
        col2=cols[1]
        s.total_steps_placeholder=col1.empty()
        s.estimated_epochs_placeholder=col2.empty()
        return

    def get_cached_models(s):
        cached_model_files=[]
        ext = '.safetensors'
        for file in os.listdir(s.settings.model_cache_folder):
            if file.endswith(ext):
                cached_model_files.append(file)
        return cached_model_files
    
    def draw_settings_widget(s):
        s.settings=omgs.OMGeneralSettings()
        widget=s.body.expander("General Settings ⚙️")
        cols=widget.columns(2)
        col1=cols[0]
        col2=cols[1]        
        s.settings.project_name=col1.text_input("Project Name", value="swwie_080120241807",placeholder="Unique logical name of your project")
        s.settings.project_dir=col2.text_input("Project Directory 📁",value="projects",placeholder="Path to project root folder")
        s.settings.model_cache_folder=col1.text_input("Model Cache Folder 📁", value="D:\Apps\stable-diffusion-webui\models\Stable-diffusion")
        s.settings.model_file=col2.selectbox("Model File", s.get_cached_models()) #text_input("Model File",value="v1-5-pruned-emaonly.safetensors")
        s.settings.save_every_n_epochs=col1.slider(label="save_every_n_epochs",min_value=1,max_value=10,value=1) # settings. #1
        s.settings.save_last_n_epochs=col2.slider(label="save_last_n_epochs",min_value=1,max_value=10,value=10) # settings. #10
        s.settings.weighted_captions=col1.checkbox(label="weighted_captions",value=False) # settings. #False
        s.settings.seed=int(col2.text_input(label="Seed",value="42")) # settings. #42
        s.settings.max_token_length=col1.slider(label="max_token_length",min_value=50,max_value=500,value=225,step=25) # settings. #225
        s.settings.v2=col2.checkbox(label="Stable Diffusion v2",value=False) # settings. #False
        s.settings.xformers=col1.checkbox(label="Use xformers",value=True) # settings. #True #True
        s.settings.lowram=col2.checkbox(label="Low RAM",value=False) # settings. #False # False
        s.settings.max_data_loader_n_workers=col1.slider(label="max_data_loader_n_workers",min_value=0,max_value=10,value=0) # settings. #0 #8
        s.settings.persistent_data_loader_workers=col2.checkbox(label="persistent_data_loader_workers",value=False) # settings. #False #True        
        s.settings.save_model_as=col1.selectbox("Save model as", ["safetensors"],index=0) # settings. #"safetensors"
        s.settings.cache_latents=col2.checkbox(label="Cache Latents",value=True) # settings. #True 
        s.settings.make_captions=col1.checkbox(label="Make Captions",value=False) # settings. #True 
        return
    
    def draw_environment_widget(s):
        s.env_widget=s.body.expander("Environment 🌳")
        return
    
    def draw_template(s):
        s.body=st.container()
        s.draw_hyper_parameter_widget()
        s.draw_settings_widget()
        s.draw_environment_widget()
        #s.draw_input_data_widget()
        s.draw_training_progress_widget()
        s.draw_training_meta_data()
        #s.draw_training_result_widget()
        return

    def main(s):
        os.system('cls')
        s.init_streamlit()
        oml.success("Started streamlit")
        s.hyper_parameters=s.build_hyper_parameters()
        s.draw_footer()
        s.draw_template()
        
        observer=omo.OMObserver(s)
        #model_callback=ommc.OMModelCallback(observer)     
        #synthetic_data_file='data/stock_prices.csv'
        #generate_synthetic_data(100, synthetic_data_file)                   
        #omt.train_model(s.hyper_parameters,observer, model_callback)
        TRAINING_IN_PROGRESS:str='TRAINING_IN_PROGRESS'
        if(TRAINING_IN_PROGRESS in st.session_state): s.training_in_progress=st.session_state[TRAINING_IN_PROGRESS]
        else: s.training_in_progress=False
        try:
            if(s.training_in_progress==False):
                if(st.button("Start training",disabled=s.training_in_progress)):
                    st.session_state[TRAINING_IN_PROGRESS]=True
                    oml.progress("training model...")
                    lora_trainer=omlt.OMLoRATrainer(observer=observer)
                    with st.spinner('Training LoRA...'): 
                        if(s.settings.make_captions): mkcap.make_captions(project_folder=s.settings.project_dir,project_name=s.settings.project_name)
                        lora_trainer.start_training(s.settings,s.hyper_parameters)                
                    oml.success("model was trained!")
                    s.training_status_ph.success("model was trained!")
            else: oml.warn("Training already in progress!")
        except():
            st.exception()
        finally:
            st.session_state[TRAINING_IN_PROGRESS]=False

app=App()
app.main()