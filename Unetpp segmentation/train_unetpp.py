# -- coding: utf-8 --
# 训练图像分割网络(unet++)模型
from Train_Unet import *
from unetpp_model import build_unetpp


def train_model_unetpp(model_type='unet++', continue_from=None):
    batch_size = BATCH_SIZE
    train_files, holdout_files = get_train_holdout_files()

    train_gen = image_generator(train_files, batch_size, True)
    holdout_gen = image_generator(holdout_files, batch_size, False)

    input_shape = (SEGMENTER_IMG_SIZE, SEGMENTER_IMG_SIZE, CHANNEL_COUNT)

    if continue_from is None:
        model = build_unetpp(input_shape=input_shape, learning_rate=1e-3, deep_supervision=True)
    else:
        model = build_unetpp(input_shape=input_shape, learning_rate=5e-4)
        model.load_weights(continue_from)

    checkpoint1 = ModelCheckpoint(
        MODEL_DIR + model_type + "_{epoch:02d}-{val_loss:.2f}.hd5", monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    dumper = DumpPredictions(holdout_files[::10], model_type)
    tensorbd = TensorBoard(log_dir='./logs', write_images=True)

    steps_per_epoch = len(train_files) // batch_size
    validation_steps = math.ceil(len(holdout_files) / batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5, patience=5,
                                  min_lr=1e-5, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=12, restore_best_weights=True)

    hist = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=80,
        validation_data=holdout_gen,
        validation_steps=validation_steps,
        callbacks=[checkpoint1, tensorbd, dumper,
                   reduce_lr, early_stop],
        verbose=1)
    import pickle
    with open("history_unetpp.pkl", "wb") as f:
        pickle.dump(hist.history, f)

    # Loss curve: use out4 only
    plt.figure(1)
    plt.plot(hist.history['out4_loss'])
    plt.plot(hist.history['val_out4_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("./temp_dir/chapter5/Unetpp_loss_curve.jpg")

    # Dice curve: use out4 only
    plt.figure(2)
    plt.plot(hist.history['out4_out4_dice'])
    plt.plot(hist.history['val_out4_out4_dice'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("./temp_dir/chapter5/Unetpp_acc_curve.jpg")

if __name__ == "__main__":
    TRAIN_LIST = './train_uid_list.csv'
    VAL_LIST = './val_uid_list.csv'
    train_model_unetpp(model_type='unet++', continue_from=None)
