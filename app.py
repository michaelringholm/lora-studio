import streamlit as st
import sys
import csv
import os
import random as rnd
import modules.om_train as omt
import modules.om_model_callback as ommc
import json
import traceback as trc
import modules.om_logging as oml
import modules.om_observer as omo
import modules.om_hyper_params as omhp
import keras

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'mymodules'))

class App():
    def __init__(s):
        oml.debug("__init__ called!")

    def init_streamlit(s):
        st.set_page_config(page_title='AI Studio', page_icon='🤖') 
        st.header('AI Studio 🤖')
        sidebar=st.sidebar
        sidebar.header('Advanced')
        #selectedAddressOption=sidebar.selectbox("Select city", options=list(options.keys()), index=st.session_state['SELECTED_HOUSE_INDEX'])
        #with st.sidebar

    def draw_city_options(s):
        if 'SELECTED_CITY_OPTION' not in st.session_state: st.session_state['SELECTED_CITY_OPTION'] = None
        sidebar=st.sidebar
        cityOptions=load_cities()
        selectedCityOption=sidebar.selectbox("Select city", options=cityOptions)
        #shuffle_city_options(sidebar,cityOptions)
        btnShuffleCityOptions=sidebar.button("🎲 Shuffle and set")
        if(btnShuffleCityOptions):
            rndCityOptionIndex=rnd.randint(1, len(cityOptions))
            st.session_state['SELECTED_CITY_OPTION']=cityOptions[rndCityOptionIndex]
        return

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

    def on_search_obsolete(s,search_term):
        oml.debug(f"searching for {search_term}")
        st.write(f"searching for {search_term}")
        searchResult=omh.searchHouses(municipalities=None,cities=search_term,limit=50)
        houses=searchResult['addresses']
        draw_houses(houses)

    def load_cities(s):
        cities=[]
        with open('../data/housing/cities.csv', 'r', encoding='utf-8') as f:
            reader=csv.reader(f)
            for row in reader:
                cities.append(row[0])
        return cities

    def draw_search_box(s):
        #global search_box_value
        #if(st.session_state['SELECTED_CITY_OPTION']!=None): search_box_value=st.session_state['SELECTED_CITY_OPTION']
        # if st.button('🔍', key='search'):
        search_term=st.text_input('City',value=search_box_value)
        radio_options = ["City", "Road"]
        selected_search_scope=st.radio("Select search scope", radio_options,horizontal=True,label_visibility='collapsed')    
        #road_search_term=st.text_input('Road',value="Mosede Kærvej")
        return search_term,selected_search_scope #,road_search_term

    def draw_buildings(s,house):
        st.markdown("## Buildings") 
        oml.success(f"{house['road']['name']} {house['houseNumber']}, {house['zip']['zipCode']} {house['zip']['name']}")
        oml.info(f" City={house['city']['name']}")
        if('buildings' in house):
            buildings=house['buildings']
            for building in buildings:
                draw_building_details(building)

    def draw_building_details(s,building):
        buildingName=f"{building['buildingName']} 🏠"
        with st.expander(buildingName, expanded=False):
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Total Area**")
                st.markdown(f"<i class='fas fa-ruler-combined'></i> {building['totalArea']} m2", unsafe_allow_html=True)            
                if 'heatingInstallation' in building: 
                    st.markdown("**Heating**") 
                    st.markdown(f"<i class='fas fa-fire'></i> {building['heatingInstallation']} 🔥", unsafe_allow_html=True)
                st.markdown("**Roofing**")
                if 'roofingMaterial' in building: st.markdown(f"<i class='fas fa-roof-raised'></i> {building['roofingMaterial']}", unsafe_allow_html=True)
            with cols[1]:                            
                if 'kitchenCondition' in building:
                    st.markdown("**Kitchen**")
                    st.markdown(f"<i class='fas fa-utensils'></i> {building['kitchenCondition']}", unsafe_allow_html=True)            
                if 'bathroomCondition' in building: 
                    st.markdown("**Bathroom**")
                    st.markdown(f"<i class='fas fa-shower'></i> {building['bathroomCondition']} 🛁", unsafe_allow_html=True)            
                if 'numberOfBathrooms' in building: 
                    st.markdown("**Number of Bathrooms**")
                    st.markdown(f"<i class='fas fa-toilet'></i> {building['numberOfBathrooms']}", unsafe_allow_html=True)
            st.markdown("---")
            cols = st.columns(2)
            with cols[0]:            
                if 'numberOfFloors' in building:
                    st.markdown("**Floors**")
                    st.markdown(f"<i class='fas fa-layer-group'></i> {building['numberOfFloors']}", unsafe_allow_html=True)             
                if 'numberOfRooms' in building:
                    st.markdown("**Rooms**")
                    st.markdown(f"<i class='fas fa-door-closed'></i> {building['numberOfRooms']} 🛏️", unsafe_allow_html=True)
            with cols[1]:     
                if 'externalWallMaterial' in building:
                    st.markdown("**Wall material**")
                    st.markdown(f"<i class='fas fa-calendar-alt'></i> {building['externalWallMaterial']} 🌇", unsafe_allow_html=True)
                if 'yearBuilt' in building:
                    st.markdown("**Year Built**")
                    st.markdown(f"<i class='fas fa-calendar-alt'></i> {building['yearBuilt']} 🗓️", unsafe_allow_html=True)
                st.markdown("**Building Name**")
                st.markdown(f"<i class='fas fa-building'></i> {buildingName}", unsafe_allow_html=True)

    def draw_sales_history(s,house):
        st.markdown("## Sales History") 
        if 'SHOW_AMOUNTS' not in st.session_state: st.session_state['SHOW_AMOUNTS'] = False
        #if(st.session_state.SHOW_AMOUNTS==True): toggleAmountsBtnLabel='Hide Amounts'
        #else: toggleAmountsBtnLabel='Show Amounts'
        col1,col2,col3 = st.columns(3)
        toggleAmountsBtnLabel="Toggle Amounts"
        with col1: toggleAmounts=st.button(toggleAmountsBtnLabel,key="ToggleAmounts")    
        if toggleAmounts:
            if(st.session_state.SHOW_AMOUNTS==True): st.session_state.SHOW_AMOUNTS=False
            else: st.session_state.SHOW_AMOUNTS=True             
        registrations=house['registrations']        
        for registration in registrations:
            draw_single_sale(registration)
            #oml.info(f" SalesPrice={formatMoney(registration['amount'])} | SalesDate={registration['date']}")

    def draw_single_sale(s,sale):        
        salesDate=sale['date']
        with st.expander(salesDate, expanded=False):
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Sales Date 🗓️**") 
                if 'date' in sale: st.markdown(f"<i class='fas fa-fire'></i> {salesDate}", unsafe_allow_html=True)
            with cols[1]:                
                st.markdown("**Sales Amount 💰**")
                if 'amount' in sale:
                    amount=sale['amount']
                    show_amounts=st.session_state.SHOW_AMOUNTS
                    if show_amounts: st.markdown(omh.formatMoney(amount))
                    else: st.markdown(f"<span style='color: grey'>{'*' * len(str(amount))}</span>", unsafe_allow_html=True)
        return

    def shuffle_address(s,houseDict):
        btnShuffleAddress=st.button("🎲 Shuffle Address")
        if(btnShuffleAddress):
            rndHouseIndex=rnd.randint(1, len(houseDict))
            oml.debug(f"Shuffling address id {rndHouseIndex}")
            #rndHouseKey = list(houseDict.keys())[rndHouseIndex]
            rndHouseKey = list(houseDict.keys())[1]
            #selectedAddressOption=rndHousevalue
            st.session_state['SELECTED_HOUSE_INDEX']=rndHouseIndex
            return

    def draw_houses(s,houses):
        options={}
        houseDict={}
        for house in houses:
            address = f"{house['road']['name']} {house['houseNumber']}, {house['zip']['zipCode']} {house['zip']['name']}"
            houseID=house['addressID']                
            options[houseID]=address
            houseDict[houseID]=house
        def print_option_labels(option):
            #oml.warn(option)
            return options[option]
        if 'SELECTED_HOUSE_INDEX' not in st.session_state: st.session_state['SELECTED_HOUSE_INDEX']=0
        shuffle_address(houseDict)
        selectedAddressOption=st.selectbox("Select address", options=list(options.keys()), format_func=print_option_labels,index=st.session_state['SELECTED_HOUSE_INDEX'])
        oml.debug(f"You selected option with houseID=[{selectedAddressOption}] and the display label [{options[selectedAddressOption]}]")
        #st.write(f"You selected option with houseID=[{selected}] and the display label [{options[selected]}]")
        selectedHouse=houseDict[selectedAddressOption]    
        draw_buildings(selectedHouse)
        draw_sales_history(selectedHouse)
        return

    def print_stack(s):
        try:
            omt.train_model()
            st.write("model was trained!")
        except Exception as ex:
            raise Exception(ex)
            st.markdown(trc.format_exception_only(ex))
            stack=trc.format_stack()
            for stackEle in stack:
                st.markdown(stackEle)

    def build_hyper_parameters(s):
        hyper_parameters=omhp.OMHyperParameters()
        hyper_parameters.batch_size=64
        hyper_parameters.num_epochs=30
        return hyper_parameters
    
    def update_training_progress(s,epoch,loss,val_loss):        
        #s.col1.write(f"epoch {epoch}")
        #s.col2.write(f"loss={loss} and val_loss={val_loss}")
        #oml.debug("update_training_progress()")
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
        s.draw_input_data_widget()
        s.draw_progress_widget()
        s.draw_training_result_widget()
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
        model_callback=ommc.OMModelCallback(observer)     
        #synthetic_data_file='data/stock_prices.csv'
        #generate_synthetic_data(100, synthetic_data_file)                   
        omt.train_model(s.hyper_parameters,observer, model_callback)
        oml.success("model was trained!")
        st.success("model was trained!")
        #st.session_state['SELECTED_HOUSE_INDEX']

app=App()
app.main()