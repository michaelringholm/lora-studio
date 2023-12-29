import modules.om_logging as oml

class OMObserver():
    DATA_LOADED_EVENT:str='data_loaded'
    MODEL_COMPILE_DONE_EVENT:str="model_compile_done"
    TRAINING_JOB_END_EVENT:str="TRAINING_JOB_END"
    EPOCH_END_EVENT:str="on_epoch_end"

    def __init__(s,app):
        s.app=app
        return
    
    def observe(s,event=None, args=None):
        #oml.debug(f"Got event {event} with #args")
        if(event!=None):
            if(event==s.EPOCH_END_EVENT):
                epoch=args[0]
                logs=args[1]
                loss=logs['loss']
                val_loss=logs['val_loss']
                s.app.update_training_progress(epoch,loss,val_loss)
            if(event==s.TRAINING_JOB_END_EVENT):
                logs=args[0]
                loss=logs['loss']
                val_loss=logs['val_loss']
                s.app.update_training_result(loss,val_loss)
            if(event==s.DATA_LOADED_EVENT):
                #train_data_features, train_data_target, eval_data_features, eval_data_target
                train_data_features=args[0]
                train_data_target=args[1]
                eval_data_features=args[2]
                eval_data_target=args[3]
                s.app.display_data(train_data_features,train_data_target,eval_data_features,eval_data_target)
        return