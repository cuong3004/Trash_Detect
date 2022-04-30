#
# # from torchmetrics.functional import accuracy
# # from torchmetrics import Precision, Recall, F1Score, Accuracy
# from model.detr import DETR
# from custom_data import CustomData
# from criterion import SetCriterion, HungarianMatcher
# import pytorch_lightning as pl
# import torch.nn as nn
# import torch
#
# from pathlib import Path
# from custom_data import CustomData
# import  torchvision.transforms as T
#
# class LitClassification(pl.LightningModule):
#     def __init__(self,
#             image_size=(320,320),
#             num_classes=4,
#             hidden_dim=256,
#             num_queries=100,
#             nheads=8,
#             dropout=0.1,
#             dim_feedforward=2048,
#             enc_layers=6,
#             dec_layers=6,
#             # pre_norm = False,
#             # deep_supervision=True,
#             ce_weight = 1.0,
#             bbox_weight = 5.0,
#             giou_weight = 2.0,
#             ):
#         super().__init__()
#
#         transformer = nn.Transformer(
#             d_model = hidden_dim, #hidden_dim
#             nhead = nheads,
#             num_encoder_layers = enc_layers,
#             num_decoder_layers = dec_layers,
#             dim_feedforward = dim_feedforward,
#             dropout = dropout,
#             activation = "gelu",
#         )
#         self.detr = DETR(num_classes=num_classes, num_queries=num_queries, transformer=transformer)
#         matcher = HungarianMatcher(cost_class = 1, cost_bbox = 5, cost_giou = 2)
#         weight_dict = {'loss_ce': ce_weight, 'loss_bbox': bbox_weight, 'loss_giou': giou_weight}
#         losses = ['labels', 'boxes', 'cardinality']
#         self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)
#
#         # outputs = model(imags)
#         # self.precision = Precision(num_classes = 4, average="macro")
#         # self._accuracy =  Accuracy(num_classes = 4, average="macro")
#         # self._f1 = F1Score(num_classes = 4, average="macro")
#         # self._recall = Recall(num_classes = 4, average="macro")
#         # self._precision = Precision(num_classes = 4, average="macro")
#
#     # def setup_metric(self):
#
#     #     self.name_states = ["train", "valid", "test"]
#     #     self.name_types = ["acc", "f1", "recall", "precision"]
#
#     #     metric_dict = {
#     #         "acc": Accuracy,
#     #         "f1": F1Score
#     #     }
#     #     for name_state in ["train", "valid", "test"]:
#     #         setattr(self, f"{name_state}_acc", Accuracy(num_classes = 4, average="macro"))
#     #         setattr(self, f"{name_state}_f1", F1Score(num_classes = 4, average="macro"))
#     #         setattr(self, f"{name_state}_recall", Recall(num_classes = 4, average="macro"))
#     #         setattr(self, f"{name_state}_precision", Precision(num_classes = 4, average="macro"))
#
#     def shared_step(self, batch, mode, **kwargs):
#         images, targets = batch
#         out = self.detr(images)
#
#         loss = self.criterion(out, targets)
#         loss = sum(loss.values())
#
#         self.log(f'{mode}_loss', loss.item(), **kwargs)
#
#         # self.log(f'{mode}_acc', self._accuracy(out, y), **kwargs)
#         # self.log(f'{mode}_f1', self._f1(out, y), **kwargs)
#         # self.log(f'{mode}_precision', self._precision(out, y), **kwargs)
#         # self.log(f'{mode}_recall', self._recall(out, y), **kwargs)
#
#         return loss
#
#     def configure_optimizers(self):
#         # optimizer = AdamW(self.parameters(),
#         #           lr = 2e-3, # args.learning_rate - default is 5e-5,
#         #           eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#         #         )
#         # total_steps = len(train_loader) * Epoch
#
#         # # Create the learning rate scheduler.
#         # scheduler = get_linear_schedule_with_warmup(optimizer,
#         #                 num_warmup_steps = 0, # Default value in run_glue.py
#         #                 num_training_steps = total_steps)
#         # return [optimizer], [scheduler]
#         return torch.optim.Adam(self.detr.parameters(), lr=0.0001)
#
#
#     def training_step(self, train_batch, batch_idx):
#
#         loss = self.shared_step(train_batch, "train", on_step=False, on_epoch=True)
#
#         return loss
#
#     def validation_step(self, val_batch, batch_idx):
#
#         self.shared_step(val_batch, "valid", on_step=False, on_epoch=True)
#
#     def test_step(self, batch, batch_idx):
#         self.shared_step(batch, "test")
#
#
# model = DETR(num_classes=4, num_queries=100)
#
#
# # k_model = pytorch_to_keras(model, input_var, [(10, 32, 32,)], verbose=True, names='short')
import torch
import numpy as np


def export_tflite(model,
                  input_sample,
                  onnx_model_path,
                  input_names=['input'],
                  output_names = ['output']):
    model.eval()
    torch.onnx.export(
        model,  # PyTorch Model
        input_sample,  # Input tensor
        onnx_model_path,  # Output file (eg. 'output_model.onnx')
        opset_version=12,  # Operator support version
        input_names= input_names,  # Input tensor name (arbitary)
        output_names = output_names,  # Output tensor name (arbitary)
    )

    import onnx
    import numpy as np

    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a Human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_model_path)
    outputs = ort_session.run(
        None,
        {'input': np.random.randn(2, 3, 320, 320).astype(np.float32)}
    )

    import onnx
    onnx_model = onnx.load(onnx_model_path)

    from onnx_tf.backend import prepare

    tf_rep = prepare(onnx_model)

    tf_model_path = "model_tf"

    tf_rep.export_graph(tf_model_path)

    import tensorflow as tf

    model = tf.saved_model.load(tf_model_path)
    model.trainable = False

    input_tensor = tf.random.uniform([2,3,320,320])
    out = model(**{'input': input_tensor})

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()

    tflite_model_path = "converted_model.tflite"
    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    import numpy as np
    import tensorflow as tf

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)