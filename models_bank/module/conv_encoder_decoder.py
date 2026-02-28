import torch.nn as nn
import torch
from .layers import general_conv2d_prenorm, fusion_prenorm_2d

BASIC_DIMS = 8

def cnn_block_3d(in_channels, out_channels, kernel_size, stride=1, padding=1, first_layer=False):
    if first_layer:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        # Example with BatchNorm
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels, momentum = 0.1, eps = 1e-5),
        )
        

def tcnn_block_3d(in_channels, out_channels, kernel_size, stride=2, padding=1, first_layer=False):
    if first_layer:
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels, momentum = 0.1, eps = 1e-5),
        )


def cnn_block(
   in_channels, out_channels, kernel_size, stride = 1, padding = 0, first_layer = False
):

   if first_layer:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size, stride = stride, padding = padding
       )
   else:
       return nn.Sequential(
           nn.Conv2d(
               in_channels, out_channels, kernel_size, stride = stride, padding = padding
           ),
           nn.BatchNorm2d(out_channels, momentum = 0.1, eps = 1e-5),
       )
       

def tcnn_block(
   in_channels,
   out_channels,
   kernel_size,
   stride = 1,
   padding = 0,
   output_padding = 0,
   first_layer = False,
):
   if first_layer:
       return nn.ConvTranspose2d(
           in_channels,
           out_channels,
           kernel_size,
           stride = stride,
           padding = padding,
           output_padding = output_padding,
       )

   else:
       return nn.Sequential(
           nn.ConvTranspose2d(
               in_channels,
               out_channels,
               kernel_size,
               stride = stride,
               padding = padding,
               output_padding = output_padding,
           ),
           nn.BatchNorm2d(out_channels, momentum = 0.1, eps = 1e-5),
       )


class Decoder_sep(nn.Module):
    def __init__(self, num_cls):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d4_c1 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*8, pad_type='reflect')
        self.d4_c2 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*8, pad_type='reflect')
        self.d4_out = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3_c1 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*4, pad_type='reflect')
        self.d3_c2 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*4, pad_type='reflect')
        self.d3_out = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2_c1 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*2, pad_type='reflect')
        self.d2_c2 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*2, pad_type='reflect')
        self.d2_out = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1_c1 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS, pad_type='reflect')
        self.d1_c2 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS, pad_type='reflect')
        self.d1_out = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv2d(in_channels=BASIC_DIMS, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, out_channels, num_modals, in_channels=BASIC_DIMS*16):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv2d_prenorm(in_channels, BASIC_DIMS*8, pad_type='reflect')
        self.d4_c2 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*8, pad_type='reflect')
        self.d4_out = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*4, pad_type='reflect')
        self.d3_c2 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*4, pad_type='reflect')
        self.d3_out = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*2, pad_type='reflect')
        self.d2_c2 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*2, pad_type='reflect')
        self.d2_out = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS, pad_type='reflect')
        self.d1_c2 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS, pad_type='reflect')
        self.d1_out = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv2d(in_channels=BASIC_DIMS*8, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv2d(in_channels=BASIC_DIMS*4, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv2d(in_channels=BASIC_DIMS*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv2d(in_channels=BASIC_DIMS, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.RFM5 = fusion_prenorm_2d(in_channel=in_channels, num_modals=num_modals)
        self.RFM4 = fusion_prenorm_2d(in_channel=BASIC_DIMS*8, num_modals=num_modals)
        self.RFM3 = fusion_prenorm_2d(in_channel=BASIC_DIMS*4, num_modals=num_modals)
        self.RFM2 = fusion_prenorm_2d(in_channel=BASIC_DIMS*2, num_modals=num_modals)
        self.RFM1 = fusion_prenorm_2d(in_channel=BASIC_DIMS*1, num_modals=num_modals)


    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv2d(in_channels=in_channels, out_channels=BASIC_DIMS, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, pad_type='reflect')
        self.e1_c3 = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, pad_type='reflect')

        self.e2_c1 = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, pad_type='reflect')
        self.e2_c3 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, pad_type='reflect')

        self.e3_c1 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, pad_type='reflect')
        self.e3_c3 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, pad_type='reflect')

        self.e4_c1 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, pad_type='reflect')
        self.e4_c3 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, pad_type='reflect')

        self.e5_c1 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*16, pad_type='reflect')
        self.e5_c3 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5