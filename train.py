# train using lightning data module and lightning module 
# and save model

import pytorch_lightning as pl

from datamodule import MyDataModule
from mybert2bert import MyB2BGenerator

def configure_callbacks():
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        verbose=True
        )

    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min',
        verbose=True,
        patience=3
        )
    return [early_stop, checkpoint]

if __name__ == '__main__':
    model = MyB2BGenerator(        
        model_save_path="saved/model.pt",
    )

    dm = MyDataModule(train_file="data/train.tsv",
                        test_file= "data/val.tsv",
                        batch_size= 8,)
    
    trainer = pl.Trainer(
            gpus=1,
            distributed_backend="ddp",
            precision=16,
            amp_backend="apex",
            max_epochs=1,
            callbacks=configure_callbacks()
        )

    trainer.fit(model, dm)



