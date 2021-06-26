import tensorflow as tf


from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.fcn import fcn_8
from keras_segmentation.models.segnet import mobilenet_segnet


#unet
unet_model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

#unet_model.train(
#    train_images =  "dataset1/images_prepped_train/",
#    train_annotations = "dataset1/annotations_prepped_train/",
#    epochs=5
#)

#unet_model.save("unet_model",save_format='h5')

unet_model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )
unet_model.load_weights("unet_model")

#unet_out = vgg_model.predict_segmentation(
#    inp="dataset1/images_prepped_test/0016E5_07965.png",
#    out_fname="D:/Downloads/vgg_out.png"
#)


print(unet_model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ))



#fcn
fcn_model = fcn_8(n_classes=51 ,  input_height=416, input_width=608  )


#fcn_model.train(
#    train_images =  "dataset1/images_prepped_train/",
#    train_annotations = "dataset1/annotations_prepped_train/",
#    epochs=5
#)

#fcn_model.save("D:/Downloads/fcn_model",save_format='h5')

fcn_model = fcn_8(n_classes=51 ,  input_height=416, input_width=608  )
fcn_model.load_weights("fcn_model")


#fcn_out = fcn_model.predict_segmentation(
#    inp="dataset1/images_prepped_test/0016E5_07965.png",
#    out_fname="D:/Downloads/fcn_out.png"
#)

print(fcn_model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ))




# Segnet
segnet_model = mobilenet_segnet(n_classes=51 ,  input_height=416, input_width=608  )

#segnet_model.train(
#    train_images =  "dataset1/images_prepped_train/",
#    train_annotations = "dataset1/annotations_prepped_train/",
#    epochs=5
#)

#segnet_model.save("segnet_model",save_format='h5')


segnet_model = mobilenet_segnet(n_classes=51 ,  input_height=416, input_width=608  )
segnet_model.load_weights("segnet_model")

#segnet_out = segnet_model.predict_segmentation(
#    inp="dataset1/images_prepped_test/0016E5_07965.png",
#    out_fname="segnet_out.png"
#)

print(segnet_model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ))





























