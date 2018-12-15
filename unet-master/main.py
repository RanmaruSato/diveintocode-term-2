from model import *
from data import *
from tensorflow.python.keras.callbacks import TensorBoard
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
myGene = trainGenerator(2,'salt/train','images','masks',data_gen_args,save_to_dir = None)

model = unet()
tsb = TensorBoard(log_dir = './logs')
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=1,epochs=1,callbacks=[model_checkpoint,tsb])

# testGene = testGenerator("data/membrane/test")
testGene = testGenerator("salt/test/images/")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("salt/test",results)



from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
#myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
myGene = trainGenerator(2,'salt/train','images','masks',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene, steps_per_epoch=300, epochs=1,callbacks=[model_checkpoint])
model.fit_generator(myGene,steps_per_epoch=2,epochs=1,callbacks=[model_checkpoint])

#testGene = testGenerator("data/membrane/test")
testGene = testGenerator("salt/test/images/")
#print("testGene□□□□□□□□□□□□□    :    ", testGene.__next__())
results = model.predict_generator(testGene,5,verbose=1)
#saveResult("/Users/tsuneo/kaggle/TGS_Salt/tomomasa/", results)
saveResult("salt/test",
           results, flag_multi_class=True)


