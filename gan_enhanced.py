'''
enhance gan model with
- add self-attention in the modality generator
- add self-attention in the modality-specific feature extractor
- add cross modal attention for fusion
- add complementary loss terms: modality-specific segmentation and multiscale fused features segmentation
'''

import torch.nn as nn
from models_bank.module.transformers import Transformer
from models_bank.module.conv_encoder_decoder import Encoder, Decoder_fuse, Decoder_sep, cnn_block, tcnn_block
import torch
import torch.nn.functional as F


BASIC_DIMS = 8
NUM_MODALS = 2
TRANSFORMER_BASIC_DIMS = 256
MLP_DIM = 512
NUM_HEADS = 8
DEPTH = 1
IMAGE_SIZE = 512
gf_dim = 32
df_dim = 64


class EnhancedGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(EnhancedGenerator, self).__init__()
        
        self.e1 = cnn_block(input_channels,gf_dim,4,2,1, first_layer = True)
        self.e2 = cnn_block(gf_dim,gf_dim*2,4,2,1,)
        self.e3 = cnn_block(gf_dim*2,gf_dim*4,4,2,1,)
        self.e4 = cnn_block(gf_dim*4,gf_dim*8,4,2,1,first_layer=True) # (batch_size, 256, 32, 32)
        
        self.patch_size = IMAGE_SIZE // 16
        self.encode_conv = nn.Conv2d(gf_dim*8, TRANSFORMER_BASIC_DIMS, kernel_size=1, stride=1, padding=0)
        self.transformer = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        self.pos = nn.Parameter(torch.zeros(1, self.patch_size**2, TRANSFORMER_BASIC_DIMS))
        
        self.d5 = tcnn_block(TRANSFORMER_BASIC_DIMS,gf_dim*4,4,2,1)
        self.d6 = tcnn_block(gf_dim*4*2,gf_dim*2,4,2,1)
        self.d7 = tcnn_block(gf_dim*2*2,gf_dim*1,4,2,1)
        self.d8 = tcnn_block(gf_dim*1*2,output_channels,4,2,1, first_layer = True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        e1 = self.e1(x)
        e2 = self.e2(F.leaky_relu(e1,0.2))
        e3 = self.e3(F.leaky_relu(e2,0.2))
        e4 = self.e4(F.leaky_relu(e3,0.2))
        
        token = self.encode_conv(e4).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
        token = self.transformer(token, self.pos)
        token = token.view(batch_size, self.patch_size, self.patch_size, TRANSFORMER_BASIC_DIMS).permute(0, 3, 1, 2).contiguous()
        
        d5 = torch.cat([self.d5(F.relu(token)),e3],1)
        d6 = torch.cat([self.d6(F.relu(d5)),e2],1)
        d7 = torch.cat([self.d7(F.relu(d6)),e1],1)
        d8 = self.d8(F.relu(d7))
        

        return self.tanh(d8)
    
    
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator,self).__init__()
        self.conv1 = cnn_block(input_channels*2,df_dim,4,2,1, first_layer=True)
        self.conv2 = cnn_block(df_dim,df_dim*2,4,2,1)# 64x64
        self.conv3 = cnn_block(df_dim*2,df_dim*4,4,2,1)# 32 x 32
        self.conv4 = cnn_block(df_dim*4,df_dim*8,4,1,1)# 31 x 31
        self.conv5 = cnn_block(df_dim*8,1,4,1,1, first_layer=True)# 30 x 30

    def forward(self, x, y):
        out = torch.cat([x,y],dim=1)
        out = F.leaky_relu(self.conv1(out),0.2)
        out = F.leaky_relu(self.conv2(out),0.2)
        out = F.leaky_relu(self.conv3(out),0.2)
        out = F.leaky_relu(self.conv4(out),0.2)
        out = self.conv5(out)

        return out


class Model(nn.Module):
    def __init__(self, num_cls, rgb_encoder_channels=3, ndsm_encoder_channels=1):
        super().__init__()
        
        self.num_cls = num_cls
        self.rgb_encoder_channels = rgb_encoder_channels
        self.ndsm_encoder_channels = ndsm_encoder_channels
        
        self.rgb_gen = EnhancedGenerator(input_channels=ndsm_encoder_channels, output_channels=rgb_encoder_channels)
        self.rgb_dis = Discriminator(input_channels=rgb_encoder_channels)
        
        # TODO this is not used in the current implementation
        # self.rgb_optimizerG = torch.optim.Adam(self.rgb_gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # self.rgb_optimizerD = torch.optim.Adam(self.rgb_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.ndsm_gen = EnhancedGenerator(input_channels=rgb_encoder_channels, output_channels=ndsm_encoder_channels)
        self.ndsm_dis = Discriminator(input_channels=ndsm_encoder_channels)

        # TODO this is not used in the current implementation
        # self.ndsm_optimizerG = torch.optim.Adam(self.ndsm_gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # self.ndsm_optimizerD = torch.optim.Adam(self.ndsm_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        self.rgb_encoder = Encoder(in_channels=rgb_encoder_channels)
        self.ndsm_encoder = Encoder(in_channels=ndsm_encoder_channels)

        # Self-attention module modality specific branch
        self.rgb_encode_conv = nn.Conv2d(BASIC_DIMS*16, TRANSFORMER_BASIC_DIMS, kernel_size=1, stride=1, padding=0)
        self.ndsm_encode_conv = nn.Conv2d(BASIC_DIMS*16, TRANSFORMER_BASIC_DIMS, kernel_size=1, stride=1, padding=0)
        
        self.patch_size = IMAGE_SIZE // 16
        self.rgb_pos = nn.Parameter(torch.zeros(1, self.patch_size**2, TRANSFORMER_BASIC_DIMS))
        self.ndsm_pos = nn.Parameter(torch.zeros(1, self.patch_size**2, TRANSFORMER_BASIC_DIMS))

        self.rgb_transformer = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        self.ndsm_transformer = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        # end of self-attention module
        
        # Dynamic fusion module 
        self.multimodal_transformer = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        self.multimodal_decode_conv = nn.Conv2d(TRANSFORMER_BASIC_DIMS*NUM_MODALS, BASIC_DIMS*16*NUM_MODALS, kernel_size=1, padding=0)
        # end of dynamic fusion module
        
        self.decoder_fuse = Decoder_fuse(out_channels=self.num_cls, num_modals=NUM_MODALS)
        
        self.decoder_sep = Decoder_sep(num_cls=self.num_cls)
        
        self.is_training = True
        
        # TODO make sure we rescale gan component loss in the final loss function
        # this weight here is simply to balance loss between bce and l1 loss
        self.L1_LAMBDA = 100
        
    def forward(self, x, masks, device='cuda'):
        batch_size = x.size(0)
        
        if self.is_training:
            rgb = x[:, :self.rgb_encoder_channels, :, :]
            ndsm = x[:, self.rgb_encoder_channels:self.rgb_encoder_channels + self.ndsm_encoder_channels, :, :].unsqueeze(1)
            
            if ndsm.dim() == 5:
                ndsm = ndsm.squeeze(1)

            # Impute RGB
            rgb_imputed = self.rgb_gen(ndsm)
            
            # Impute NDSM
            ndsm_imputed = self.ndsm_gen(rgb)
            
            # Real rgb and ndsm signals
            rgb_real_x1, rgb_real_x2, rgb_real_x3, rgb_real_x4, rgb_real_x5 = self.rgb_encoder(rgb)
            ndsm_real_x1, ndsm_real_x2, ndsm_real_x3, ndsm_real_x4, ndsm_real_x5 = self.ndsm_encoder(ndsm)
            
            # assume there is consistent mask for each batch
            full_mask = torch.tensor([True, True]).to(device)
            miss_ndsm = torch.tensor([True, False]).to(device)
            miss_rgb = torch.tensor([False, True]).to(device)
            
            if (torch.equal(masks[0], full_mask)):
                # print('full')
                rgb_token_x5 = self.rgb_encode_conv(rgb_real_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
                ndsm_token_x5 = self.ndsm_encode_conv(ndsm_real_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
                rgb_intra_token_x5 = self.rgb_transformer(rgb_token_x5, self.rgb_pos)
                ndsm_intra_token_x5 = self.ndsm_transformer(ndsm_token_x5, self.ndsm_pos)
                
                rgb_pred = self.decoder_sep(rgb_real_x1, rgb_real_x2, rgb_real_x3, rgb_real_x4, rgb_real_x5)
                ndsm_pred = self.decoder_sep(ndsm_real_x1, ndsm_real_x2, ndsm_real_x3, ndsm_real_x4, ndsm_real_x5)
                
                x1_features = torch.cat((rgb_real_x1, ndsm_real_x1), dim=1)
                x2_features = torch.cat((rgb_real_x2, ndsm_real_x2), dim=1)
                x3_features = torch.cat((rgb_real_x3, ndsm_real_x3), dim=1)
                x4_features = torch.cat((rgb_real_x4, ndsm_real_x4), dim=1)

                ndsm_d_loss = 0.0
                rgb_d_loss = 0.0
                ndsm_g_loss = 0.0
                rgb_g_loss = 0.0
                
            elif (torch.equal(masks[0], miss_ndsm)):
                # print('Imputing NDSM')
                ndsm_d_real = self.ndsm_dis(ndsm, ndsm)
                ndsm_d_real_loss = self.bce_loss(ndsm_d_real, torch.ones_like(ndsm_d_real))
                ndsm_d_fake = self.ndsm_dis(ndsm, ndsm_imputed.detach())
                ndsm_d_fake_loss = self.bce_loss(ndsm_d_fake, torch.zeros_like(ndsm_d_fake))
                ndsm_d_loss = (ndsm_d_real_loss + ndsm_d_fake_loss) * 0.5
                # train Generator
                ndsm_g_fake = self.ndsm_dis(ndsm, ndsm_imputed)
                ndsm_g_loss = self.bce_loss(ndsm_g_fake, torch.ones_like(ndsm_g_fake))
                ndsm_l1_loss = self.l1_loss(ndsm_imputed, ndsm) * self.L1_LAMBDA
                ndsm_g_loss = ndsm_g_loss + ndsm_l1_loss

                rgb_d_loss = 0.0
                rgb_g_loss = 0.0
            
                ndsm_imputed_x1, ndsm_imputed_x2, ndsm_imputed_x3, ndsm_imputed_x4, ndsm_imputed_x5 = self.ndsm_encoder(ndsm_imputed)
                
                rgb_token_x5 = self.rgb_encode_conv(rgb_real_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
                ndsm_token_x5 = self.ndsm_encode_conv(ndsm_imputed_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
                rgb_intra_token_x5 = self.rgb_transformer(rgb_token_x5, self.rgb_pos)
                ndsm_intra_token_x5 = self.ndsm_transformer(ndsm_token_x5, self.ndsm_pos)
                
                rgb_pred = self.decoder_sep(rgb_real_x1, rgb_real_x2, rgb_real_x3, rgb_real_x4, rgb_real_x5)
                ndsm_pred = self.decoder_sep(ndsm_imputed_x1, ndsm_imputed_x2, ndsm_imputed_x3, ndsm_imputed_x4, ndsm_imputed_x5)
            
                x1_features = torch.cat((rgb_real_x1, ndsm_imputed_x1), dim=1)
                x2_features = torch.cat((rgb_real_x2, ndsm_imputed_x2), dim=1)
                x3_features = torch.cat((rgb_real_x3, ndsm_imputed_x3), dim=1)
                x4_features = torch.cat((rgb_real_x4, ndsm_imputed_x4), dim=1)
                
            elif (torch.equal(masks[0], miss_rgb)):
                # print('Imputing RGB')

                # Train Discriminator
                rgb_d_real = self.rgb_dis(rgb, rgb)
                rgb_d_real_loss = self.bce_loss(rgb_d_real, torch.ones_like(rgb_d_real))
                rgb_d_fake = self.rgb_dis(rgb, rgb_imputed.detach())
                rgb_d_fake_loss = self.bce_loss(rgb_d_fake, torch.zeros_like(rgb_d_fake))
                rgb_d_loss = (rgb_d_real_loss + rgb_d_fake_loss) * 0.5
                # train Generator
                rgb_g_fake = self.rgb_dis(rgb, rgb_imputed)
                rgb_g_loss = self.bce_loss(rgb_g_fake, torch.ones_like(rgb_g_fake))
                rgb_l1_loss = self.l1_loss(rgb_imputed, rgb) * self.L1_LAMBDA
                rgb_g_loss = rgb_g_loss + rgb_l1_loss

                rgb_imputed_x1, rgb_imputed_x2, rgb_imputed_x3, rgb_imputed_x4, rgb_imputed_x5 = self.rgb_encoder(rgb_imputed)
                
                rgb_token_x5 = self.rgb_encode_conv(rgb_imputed_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
                ndsm_token_x5 = self.ndsm_encode_conv(ndsm_real_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
                rgb_intra_token_x5 = self.rgb_transformer(rgb_token_x5, self.rgb_pos)
                ndsm_intra_token_x5 = self.ndsm_transformer(ndsm_token_x5, self.ndsm_pos)
                
                rgb_pred = self.decoder_sep(rgb_imputed_x1, rgb_imputed_x2, rgb_imputed_x3, rgb_imputed_x4, rgb_imputed_x5)
                ndsm_pred = self.decoder_sep(ndsm_real_x1, ndsm_real_x2, ndsm_real_x3, ndsm_real_x4, ndsm_real_x5)
                
                x1_features = torch.cat((rgb_imputed_x1, ndsm_real_x1), dim=1)
                x2_features = torch.cat((rgb_imputed_x2, ndsm_real_x2), dim=1)
                x3_features = torch.cat((rgb_imputed_x3, ndsm_real_x3), dim=1)
                x4_features = torch.cat((rgb_imputed_x4, ndsm_real_x4), dim=1)

                ndsm_d_loss = 0.0
                ndsm_g_loss = 0.0
            
            
            # Dynamic fusion module
            multimodal_token_x5 = torch.cat((rgb_intra_token_x5, ndsm_intra_token_x5), dim=1)
            multimodal_pos = torch.cat((self.rgb_pos, self.ndsm_pos), dim=1)
            multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
            multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), self.patch_size, self.patch_size, TRANSFORMER_BASIC_DIMS*NUM_MODALS).permute(0, 3, 1, 2).contiguous())
            fuse_preds, preds = self.decoder_fuse(x1_features, x2_features, x3_features, x4_features, multimodal_inter_x5)
            

            dict_results = {
                'rgb_g_loss': rgb_g_loss,
                'rgb_d_loss': rgb_d_loss,
                'ndsm_g_loss': ndsm_g_loss,
                'ndsm_d_loss': ndsm_d_loss,
                'rgb_preds': rgb_pred,
                'ndsm_preds': ndsm_pred,
                'fuse_preds': fuse_preds,
                'fuse_scale_preds': preds,
            }
            
        else:
            # Inference mode
            miss_ndsm = torch.tensor([True, False]).to(device)
            miss_rgb = torch.tensor([False, True]).to(device)
            
            for i in range(x.shape[0]):
                if (torch.equal(masks[i], miss_ndsm)):
                    # print('Imputing NDSM')
                    x[i, self.rgb_encoder_channels:self.rgb_encoder_channels+self.ndsm_encoder_channels, :, :] = self.ndsm_gen(x[i, :self.rgb_encoder_channels, :, :].unsqueeze(0))
                elif (torch.equal(masks[i], miss_rgb)):
                    # print('Imputing RGB')
                    x[i, :self.rgb_encoder_channels, :, :] = self.rgb_gen(x[i, self.rgb_encoder_channels:self.rgb_encoder_channels+self.ndsm_encoder_channels, :, :].unsqueeze(0))

            rgb_x1, rgb_x2, rgb_x3, rgb_x4, rgb_x5 = self.rgb_encoder(x[:, :self.rgb_encoder_channels, :, :])
            
            if self.ndsm_encoder_channels >= 2:
                ndsm_x1, ndsm_x2, ndsm_x3, ndsm_x4, ndsm_x5 = self.ndsm_encoder(x[:, self.rgb_encoder_channels:self.rgb_encoder_channels+self.ndsm_encoder_channels, :, :])
            else:
                ndsm_x1, ndsm_x2, ndsm_x3, ndsm_x4, ndsm_x5 = self.ndsm_encoder(x[:, self.rgb_encoder_channels:self.rgb_encoder_channels+self.ndsm_encoder_channels, :, :].unsqueeze(1))

            rgb_token_x5 = self.rgb_encode_conv(rgb_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
            ndsm_token_x5 = self.ndsm_encode_conv(ndsm_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
            rgb_intra_token_x5 = self.rgb_transformer(rgb_token_x5, self.rgb_pos)
            ndsm_intra_token_x5 = self.ndsm_transformer(ndsm_token_x5, self.ndsm_pos)
            
            x1_features = torch.cat((rgb_x1, ndsm_x1), dim=1)
            x2_features = torch.cat((rgb_x2, ndsm_x2), dim=1)
            x3_features = torch.cat((rgb_x3, ndsm_x3), dim=1)
            x4_features = torch.cat((rgb_x4, ndsm_x4), dim=1)
            
            multimodal_token_x5 = torch.cat((rgb_intra_token_x5, ndsm_intra_token_x5), dim=1)
            multimodal_pos = torch.cat((self.rgb_pos, self.ndsm_pos), dim=1)
            multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
            multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), self.patch_size, self.patch_size, TRANSFORMER_BASIC_DIMS*NUM_MODALS).permute(0, 3, 1, 2).contiguous())
            
            fuse_preds, preds = self.decoder_fuse(x1_features, x2_features, x3_features, x4_features, multimodal_inter_x5)
            
            dict_results = {
                'fuse_preds': fuse_preds
            }
            
        return dict_results