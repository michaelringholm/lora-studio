import modules.om_logging as oml

class OMObserver():
    TRAINING_STEP_EVENT:str='TRAINING_STEP_EVENT'
    TRANING_START_VALIDATE_EVENT:str='TRANING_START_VALIDATE_EVENT'
    CUDA_INFO_EVENT:str='CUDA_INFO_EVENT'
    CACHED_LATENT_EVENT:str='CACHED_LATENT_EVENT'

    def __init__(s,app):
        s.app=app
        return
    
    def observe(s,event=None, args=None):
        #oml.debug(f"Got event {event} with #args")
        if(event!=None):
            if(event==s.TRAINING_STEP_EVENT):
                epoch=args[0]
                step=args[1]
                current_loss=args[2]
                avr_loss=args[3]
                global_step=args[4]
                #oml.debug(f"batch={batch}")
                #oml.debug(f"global_step={global_step}")
                s.app.update_training_progress(epoch,step,current_loss,avr_loss,global_step)
            elif(event==s.TRANING_START_VALIDATE_EVENT):
                pre_steps_per_epoch=args[0]
                steps_per_epoch=args[1]
                total_steps=args[2]
                estimated_epochs=args[3]
                s.app.update_training_start_meta_data(pre_steps_per_epoch,steps_per_epoch,total_steps,estimated_epochs)
            elif(event==s.CUDA_INFO_EVENT):
                gpu=args[0]
                cuda_installed=args[1]
                s.app.update_cuda_info(gpu,cuda_installed)
            elif(event==s.CACHED_LATENT_EVENT):
                batch=args[0]
                num_batches=args[1]
                next_image_batch=args[2]
                s.app.update_cache_latents_progress(batch,num_batches,next_image_batch)
        return