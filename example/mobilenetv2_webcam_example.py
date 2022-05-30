# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# based on the MobileNetv2 implementation from PyTorch
# source: https://pytorch.org/vision/0.8/_modules/torchvision/models/mobilenet.html
# and: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

# original license and copyright:
# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


########################################################################################################
#                                    MobileNetv2 Webcam example
#
# This is a very simple example that shows how DeltaCNN can be used as a replacement for torch.nn layers.
# Please take a look at the changes made to mobilenet for the deltacnn version.
# All changes are marked with a '# added' or '# replaced by' comment.
#  
# This example uses weights pretrained on ImageNet. 
# Also, the webcam is used as video input to avoid having to download videos and for being able to play 
# around with the camera.
# Adjust the delta_threshold to see how it affects the predictions.
#
########################################################################################################



import torch
from torch import nn
from deltacnn.sparse_layers import DCBackend, DCConv2d, DCThreshold
from mobilenet_original import mobilenet_v2
from mobilenet_deltacnn import DeltaCNN_mobilenet_v2

def test():
    from PIL import Image
    from torchvision import transforms
    import cv2

    device="cuda:0"

    original_model = mobilenet_v2(pretrained=True, progress=True)
    original_model.eval()
    original_model.to(device, memory_format=torch.channels_last)
    dc_model = DeltaCNN_mobilenet_v2(pretrained=True, progress=True)
    dc_model.eval()
    dc_model.to(device, memory_format=torch.channels_last)
    dc_model.process_filters()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Read the categories
    with open("example/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    camera = cv2.VideoCapture(0)

    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)

    while True:
        ret, input_image = camera.read()
        if not ret:
            break

        # cv2.imshow("cam", input_image)
        # if cv2.waitKey(1) == 27: 
        #     break  # esc to quit

        input_tensor = preprocess(Image.fromarray(input_image))
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # increase batch size to increase workload
        input_batch = torch.repeat_interleave(input_batch, 32, dim=0)

        # move the input and model to GPU for speed if available
        input_batch = input_batch.to(device).contiguous(memory_format=torch.channels_last)

        torch.cuda.synchronize()
        with torch.no_grad():
            time_start.record()
            original_output = original_model(input_batch)
            time_end.record()
        torch.cuda.synchronize()
        duration_original = time_start.elapsed_time(time_end)

        with torch.no_grad():
            time_start.record()
            dc_output = dc_model(input_batch)
            time_end.record()
        torch.cuda.synchronize()
        duration_dc = time_start.elapsed_time(time_end)

        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(dc_output[0], dim=0)
        # print(probabilities)

        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        print("\r", end="")
        for i in range(top5_prob.size(0)):
            print(f"{categories[top5_catid[i]]:<16} {top5_prob[i].item():.3f} ", end="")
        
        print(f"original: {duration_original:.2f}ms, dc: {duration_dc:.2f}ms out_diff_mean={(dc_output[0]-original_output[0]).abs().mean():.3f}     ", end="")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("cuda not available. example.py requires cuda")
        exit(-1)

    # using a low default threshold of 0.05. play around with this value to see how it affects performance and accuracy.
    DCThreshold.t_default = 0.05
    DCConv2d.backend = DCBackend.deltacnn
    test()