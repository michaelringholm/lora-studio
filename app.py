import streamlit as st
import sys
import csv
import os
import random as rnd
import modules.om_lora_trainer as omlt
import json
import traceback as trc
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
    
    def update_training_progress(s,epoch,step,current_loss,avg_loss):
        total_steps=int(s.total_steps)
        progress=(step)/total_steps # s.hyper_parameters.num_epochs*20
        s.training_progress_bar.progress(progress)
        s.progress_bar_text.markdown(f"epoch {step+1}/{total_steps}, current_loss={current_loss}, avg_loss={avg_loss}") # TODO use steps s.hyper_parameters.num_epochs
        s.current_loss_data.append({"step":step, "current_loss":current_loss})    
        s.avg_loss_data.append({"step":step, "avg_loss":avg_loss})
        s.current_loss_line_chart_placeholder.line_chart(s.current_loss_data,x="step",y="current_loss")
        s.avg_loss_line_chart_placeholder.line_chart(s.avg_loss_data,x="step",y="avg_loss")
        return
    
    def update_training_start_meta_data(s,pre_steps_per_epoch,steps_per_epoch,total_steps,estimated_epochs):
        s.total_steps=s.total_steps_placeholder.text_input("Total Steps",value=total_steps)
        s.estimated_epochs_placeholder.text_input("Estimated Epocs",value=estimated_epochs)
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
        s.training_progress_bar=widget.progress(value=0,text='Step')
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
        widget=s.body.expander("Hyper Parameters")
        cols=widget.columns(2)
        col1=cols[0]
        col2=cols[1]
        s.hyper_parameters.max_epochs=col1.slider(label="Epochs",min_value=1,max_value=200,value=1)
        s.hyper_parameters.batch_size=col1.slider(label="Batch Size",min_value=1,max_value=128,value=1)
        display_factor=10000
        s.hyper_parameters.learning_rate=(col1.slider(label="Learning Rate",min_value=1,max_value=100,step=5,value=10)/display_factor)
        s.hyper_parameters.first_layer_neurons=col2.slider(label="First Layer Neurons",min_value=1,max_value=1000,value=30)
        s.hyper_parameters.hidden_layers=col2.slider(label="Hidden Layers",min_value=1,max_value=10,value=2)
        s.hyper_parameters.output_nodes=col2.slider(label="Output Nodes",min_value=1,max_value=10,value=1)   
        s.hyper_parameters.num_image_repeats=col1.slider(label="Image Repeats",value=5,min_value=1,max_value=20)          
        s.hyper_parameters.clip_skip=col2.slider(label="Clip Skip",value=2,min_value=1,max_value=4)
        s.hyper_parameters.optimizer_name=widget.selectbox("Optimizer", ["Adam","Adamax","AdamW","Adagrad"],index=0)
        return
    
    def draw_training_meta_data(s):
        widget=s.body.expander("Training Meta Data")
        cols=widget.columns(2)
        col1=cols[0]
        col2=cols[1]
        s.total_steps_placeholder=col1.empty()
        s.estimated_epochs_placeholder=col2.empty()
        return

    def draw_settings_widget(s):
        s.settings=omgs.OMGeneralSettings()
        widget=s.body.expander("General Settings")
        #project_name = "pernille_harder_211220231821"        
        #model_file="v1-5-pruned-emaonly.safetensors" # cyberrealistic_v40.safetensors | realcartoon3d_v8.safetensors" | "realcartoonRealistic_v11.safetensors" | aZovyaPhotoreal_v2.safetensors"
        #model_url=f"https://xyz.com/{s.model_file}"
        #s.settings.model_file=widget.text_input("Model File", value="v1-5-pruned-emaonly.safetensors")
        #oml.debug(f"mod_fi={s.settings.model_file}")
        s.settings.project_name=widget.text_input("Project Name", value="albbukjor_301220230945",placeholder="Unique logical name of your project")
        s.settings.project_dir=widget.text_input("Project Directory",value="projects",placeholder="Path to project root folder")
        s.settings.model_cache_folder=widget.text_input("Model Cache Folder", value="D:\Apps\stable-diffusion-webui\models\Stable-diffusion")
        s.settings.model_file=widget.text_input("Model File",value="v1-5-pruned-emaonly.safetensors")  
        return
    
    def draw_template(s):
        s.body=st.container()
        s.draw_hyper_parameter_widget()
        s.draw_settings_widget()
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
        oml.progress("training model...")
        observer=omo.OMObserver(s)
        #model_callback=ommc.OMModelCallback(observer)     
        #synthetic_data_file='data/stock_prices.csv'
        #generate_synthetic_data(100, synthetic_data_file)                   
        #omt.train_model(s.hyper_parameters,observer, model_callback)
        lora_trainer=omlt.OMLoRATrainer(observer=observer)
        try:
            #lora_trainer.start_training(project_name=s.settings.project_name,project_dir="projects",batch_size=s.hyper_parameters.batch_size, 
             #                           max_epochs=s.hyper_parameters.max_epochs,model_file=s.settings.model_file,model_cache_folder=s.settings.model_cache_folder,
              #                          num_image_repeats=s.hyper_parameters.num_image_repeats)
            lora_trainer.start_training(s.settings,s.hyper_parameters)
            oml.success("model was trained!")
            st.success("model was trained!")
        except():
            st.exception()

app=App()
app.main()