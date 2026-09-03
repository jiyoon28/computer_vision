# U-Net
# Input -> Encoder -> Bottleneck -> Decoder <- Skip Connection -> 1x1 Conv -> Prediction Mask
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    U-Net에서 반복해서 사용하는 기본 블록
    Conv 3x3
    ↓
    BatchNorm
    ↓
    ReLU
    ↓
    Conv 3x3
    ↓
    BatchNorm
    ↓
    ReLU
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # =========================
        # Encoder
        # =========================
        
        # 1x256x256 -> 32x256x256
        self.enc1=DoubleConv(in_channels, 32)
        self.pool1=nn.MaxPool2d(2)
        
        # 32x128x128 -> 64x128x128
        self.enc2=DoubleConv(32,64)
        self.pool2=nn.MaxPool2d(2)

        # 64x64x64 -> 128x64x64
        self.enc3=DoubleConv(64,128)
        self.pool3=nn.MaxPool2d(2)
        
        # 128x32x32 -> 256x32x32
        self.enc4=DoubleConv(128,256)
        self.pool4=nn.MaxPool2d(2)
        
        # =========================
        # Bottleneck
        # =========================
        
        # 256x16x16 -> 512x16x16
        self.bottleneck=DoubleConv(256,512)
        
        # =========================
        # Decoder
        # =========================
        # 16x16 -> 32x32
        # 512 -> 256 channels
        self.up4=nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        # Decoder 256 + Encoder 256 Concatenate -> 512
        self.dec4=DoubleConv(512,256)
        
        # 32x32 -> 64x64
        self.up3=nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        
        # Decoder 128 + Encoder 128 = 256
        self.dec3 = DoubleConv(256,128)
        
        # 64x64 -> 128x128
        self.up2=nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Decoder 64 + Encoder 64 = 128
        self.dec2=DoubleConv(128, 64)
        
        # 128x128 -> 256x256
        self.up1=nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # Decoder 32 + Encoder 32 = 64
        self.dec1=DoubleConv(64, 32)
        
        # =========================
        # Final 1x1 Conv
        # =========================
        
        # 각 Pixel 마다 존재하는 32개의 feature을 defect score 1개로 변환
        # 32x256x256 -> 1x256x256
        self.final=nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self,x):
        # =========================
        # Encoder
        # =========================
        
        # 1x256x256 -> 32x256x256
        e1=self.enc1(x)
        
        # 32x256x256 -> Pool -> 32x128x128 -> Conv -> 64x128x128
        e2=self.enc2(self.pool1(e1))
        
        # 64x128x128 -> 128x64x64
        e3=self.enc3(self.pool2(e2))
        
        # 128x64x64 -> 256x32x32
        e4=self.enc4(self.pool3(e3))
        
        # =========================
        # Bottleneck
        # =========================
        
        # U-Net에서 가장 깊은 지점
        # 해상도는 가장 작고 channel은 가장 많다
        # 256x32x32 -> Pool -> 256x16x16 -> Conv -> 512x16x16
        b=self.bottleneck(self.pool4(e4))
        
        # =========================
        # Decoder 4
        # =========================
        
        # 512x16x16 -> 256x32x32
        d4=self.up4(b)
        
        # U-Net Skip Conncecion
        # Decoder feature 256x32x32
        # Encoder feature e4 256x32x32
        # Channel 방향으로 concatenate ->  512x32x32
        d4=torch.cat([d4,e4],dim=1)
        
        # 512 -> 256
        d4=self.dec4(d4)
        
        # =========================
        # Decoder 3
        # =========================
        
        d3=self.up3(d4)
        
        # 128 + 128
        d3=torch.cat([d3,e3],dim=1)
        d3=self.dec3(d3)
        
        # =========================
        # Decoder 2
        # =========================
        d2=self.up2(d3)
        
        # 64 + 64
        d2=torch.cat([d2,e2],dim=1)
        d2=self.dec2(d2)
        
        # =========================
        # Decoder 1
        # =========================
        
        d1=self.up1(d2)
        
        # 32 + 32
        d1=torch.cat([d1,e1],dim=1)
        d1=self.dec1(d1)
        
        # =========================
        # Final Output
        # =========================
        
        # sigmoid 전의 raw score = logits
        logits = self.final(d1)
        return logits
    
# 모델 Shape Test
if __name__=="__main__":
    model=UNet(
        in_channels=1, out_channels=1
    )
    
    # Batch=2, Channel=1, Height=256, Width=2565
    x=torch.randn(2,1,256,256)
    y=model(x)
    
    print("Input Shape :", x.shape)
    print("Output Shape: ", y.shape)