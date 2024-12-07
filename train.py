from Trainer import Trainer




def main():
    # Create the model trainer
    batch_size=256 # Total batch size over all GPUs (so if 2 GPUs and batch_size=256, each GPU will see a batch size of 128)
    learning_rate=1e-4
    warmup_steps=10_000
    num_steps=1_000_000
    dev="gpu"
    wandb_name="test"
    project_name="test"
    log_steps=10
    use_amp=True
    attention_type="eager" # eager, flash_attention_2, sdpa, custom
    clipping_value=None
    weight_decay=0.01
    model_save_path = "models/test"
    num_save_steps = 1_000
    keep_dataset_in_mem = False
    model_max_length = 1024

    # dataset = "TrevorDohm/Pile_TokLlama"
    dataset = "gmongaras/dummy_text_dataset"
    cache_path = "/users/gmongaras/work/datasets/data_cache/"
    
    # Load in a checkpoint
    load_checkpoint = False
    checkpoint_path = "models/test/"
    
    trainer = Trainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_steps=num_steps,
        dev=dev,
        wandb_name=wandb_name,
        project_name=project_name,
        log_steps=log_steps,
        use_amp=use_amp,
        attention_type=attention_type,
        clipping_value=clipping_value,
        weight_decay=weight_decay,
        model_save_path=model_save_path,
        num_save_steps=num_save_steps,
        keep_dataset_in_mem=keep_dataset_in_mem,
        load_checkpoint=load_checkpoint,
        checkpoint_path=checkpoint_path,
        model_max_length=model_max_length,
        cache_path=cache_path,
    )
    
    # Train model
    trainer(dataset)





if __name__ == "__main__":
    main()
