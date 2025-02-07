import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn





class SpenerNet(nn.Module):
    # 图像尺寸不改变然后padding网络大小

    def __init__(self, encoder_config, network_config, feature_dim=32):

        super(SpenerNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, feature_dim, 3, padding=1)
        )

        coor_encoder_out_dim = encoder_config['n_levels'] * encoder_config['n_features_per_level']

        self.coord_encoder = tcnn.Encoding(n_input_dims=2, encoding_config=encoder_config)


        self.intensity_decoder = tcnn.Network(n_input_dims=feature_dim + coor_encoder_out_dim,
                                    n_output_dims=1, network_config=network_config)


    def forward(self, img, xy):
        # encoding feature
        feature_vector = self.encoder(img)
        bs, feature_dim, H, W = feature_vector.shape
        bs, sample_num, length, _ = xy.shape
        xy = xy.reshape(-1, 2)
        pad_left = (length - H)//2 
        pad_right = length - H - pad_left
        pad_top = (length - W) // 2
        pad_bottom = length - W - pad_top
        feature_vector = F.pad(feature_vector, (pad_left, pad_right, pad_top, pad_bottom), mode = 'replicate')

        
        select_idx = xy.flip(1).unsqueeze(0).unsqueeze(0)
        coord_vector = self.coord_encoder(xy)
        select_vector = F.grid_sample(feature_vector, select_idx, mode='bilinear', padding_mode='zeros', align_corners=True)
        select_vector = torch.permute(select_vector, (0, 2, 3, 1)).reshape(-1, feature_dim)
        decoder_in = torch.cat([select_vector, coord_vector], dim=1)
        pre_intensity = self.intensity_decoder(decoder_in)

        return pre_intensity

class Encoder(nn.Module):
    
        def __init__(self, feature_dim=32):
    
            super(Encoder, self).__init__()
    
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 48, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, feature_dim, 3, padding=1)
            )
    
    
    
        def forward(self, img):
            # encoding feature
            feature_vector = self.encoder(img)
            return feature_vector
        
class Decoder(nn.Module):
    
        def __init__(self, encoder_config, network_config, feature_dim=32):
    
            super(Decoder, self).__init__()
    
            coor_encoder_out_dim = encoder_config['n_levels'] * encoder_config['n_features_per_level']
    
            self.coord_encoder = tcnn.Encoding(n_input_dims=2, encoding_config=encoder_config)
            self.feature_dim = feature_dim
    
            self.intensity_decoder = tcnn.Network(n_input_dims=feature_dim + coor_encoder_out_dim,
                                        n_output_dims=1, network_config=network_config)
    
    
        def forward(self, feature_vector, xy):
            # flip the coordinates xy shape (-1, 2)
            select_idx = xy.flip(1).unsqueeze(0).unsqueeze(0)
            coord_vector = self.coord_encoder(xy)
            select_vector = F.grid_sample(feature_vector, select_idx, mode='bilinear', padding_mode='zeros', align_corners=True)
            select_vector = torch.permute(select_vector, (0, 2, 3, 1)).reshape(-1, self.feature_dim)
            decoder_in = torch.cat([select_vector, coord_vector], dim=1)
            pre_intensity = self.intensity_decoder(decoder_in)
    
            return pre_intensity
        

def pad_feature(feature_vector, length):
    bs, feature_dim, H, W = feature_vector.shape
    pad_left = (length - H)//2 
    pad_right = length - H - pad_left
    pad_top = (length - W) // 2
    pad_bottom = length - W - pad_top
    feature_vector = F.pad(feature_vector, (pad_left, pad_right, pad_top, pad_bottom), mode = 'replicate')
    return feature_vector

