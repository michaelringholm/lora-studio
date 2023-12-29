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
        with open('./css/custom.css') as f:
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
    
    def update_training_progress(s,epoch,loss,val_loss):        
        progress=(epoch+1)/s.hyper_parameters.num_epochs
        s.training_progress_bar.progress(progress)
        s.progress_bar_text.markdown(f"epoch {epoch+1}/{s.hyper_parameters.num_epochs}")
        s.loss_data.append({"epoch":epoch, "loss":loss})        
        s.lc.line_chart(s.loss_data,x="epoch",y="loss")
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

    def draw_progress_widget(s):
        widget=s.body.container(border=True)
        widget.subheader("Training progress")
        s.training_progress_bar=widget.progress(value=0,text='Epoch')
        s.progress_bar_text=widget.empty()
        s.loss_data=[{"epoch":None, "loss":None}]
        #s.loss_data.append({"epoch":1, "loss":6.25})
        #s.loss_data.append({"epoch":2, "loss":8.25})
        #s.loss_data[1]="8.65"
        #s.loss_line_chart=widget.line_chart(s.loss_data,x="epoch",y="loss")
        #s.progress_widget=widget
        s.lc=widget.empty() #widget.line_chart(s.loss_data,x="epoch",y="loss")
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
        s.hyper_parameters.num_epochs=col1.slider(label="Epochs",min_value=1,max_value=200,value=20)
        s.hyper_parameters.batch_size=col1.slider(label="Batch Size",min_value=1,max_value=128,value=1)
        s.hyper_parameters.learning_rate=col1.slider(label="Learning Rate",min_value=1e-6,max_value=1e-2,value=1e-3)
        s.hyper_parameters.first_layer_neurons=col2.slider(label="First Layer Neurons",min_value=1,max_value=1000,value=30)
        s.hyper_parameters.hidden_layers=col2.slider(label="Hidden Layers",min_value=1,max_value=10,value=2)
        s.hyper_parameters.output_nodes=col2.slider(label="Output Nodes",min_value=1,max_value=10,value=1)
        widget.selectbox("Optimizer", ["Adam","Adamax","AdamW","Adagrad"])
        return
    
    def draw_template(s):
        s.body=st.container(border=False)
        s.draw_hyper_parameter_widget()
        #s.draw_input_data_widget()
        #s.draw_progress_widget()
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
        lora_trainer=omlt.OMLoRATrainer()
        try:
            lora_trainer.start_training(project_name="pernille_harder_211220231821",project_dir="projects")
            oml.success("model was trained!")
            st.success("model was trained!")
        except():
            st.exception()

app=App()
app.main()