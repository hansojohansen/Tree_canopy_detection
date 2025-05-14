import torch
import torch.nn as nn
import torch.nn.functional as F

# Modul for to konvolusjonslag med Batch Normalization (BN) og ReLU-aktivisering
class DoubleConv(nn.Module):
    """(konvolusjon => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        in_channels: Antall kanaler i input (for RGB = 3)
        out_channels: Antall kanaler i output
        mid_channels: Antall kanaler for mellomlag. Settes til out_channels hvis ikke spesifisert.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # Definerer et sekvensielt lag med konvolusjon, batch normalisering, og ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Forward pass gjennom dobbelt konvolusjonslag
        return self.double_conv(x)

# Modul for nedskalering av bildedata med maks-pooling etterfulgt av DoubleConv
class Down(nn.Module):
    """Nedskalering med maxpool etterfulgt av dobbelt konvolusjon"""
    def __init__(self, in_channels, out_channels):
        """
        in_channels: Antall kanaler i input
        out_channels: Antall kanaler i output
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Halverer oppløsningen i H og W
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # Forward pass gjennom maxpooling og deretter DoubleConv
        return self.maxpool_conv(x)

# Modul for oppskalering av bildedata, med mulighet for bilineær interpolasjon
class Up(nn.Module):
    """Oppskalering etterfulgt av dobbelt konvolusjon"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        """
        in_channels: Antall kanaler i input
        out_channels: Antall kanaler i output
        bilinear: Boolsk, bruk bilineær oppskalering hvis True, ellers bruk ConvTranspose2d
        """
        super().__init__()
        # Bilineær interpolasjon brukes som standard for oppskalering
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Alternativ: bruk ConvTranspose2d for oppskalering
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        # Oppskalering av x1 og tilpasning av dimensjoner før sammenslåing
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # Justerer dimensjonene med padding for nøyaktig samsvar
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # Sammenslåing av oppskalert og tilsvarende input
        return self.conv(x)  # Forward gjennom dobbelt konvolusjonslag

# Modul for å generere endelig output med spesifisert antall utkanaler (f.eks. antall klasser)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: Antall kanaler i input
        out_channels: Antall kanaler i output (f.eks. antall klasser i segmenteringen)
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Bruker kjerne på 1x1 for å redusere antall kanaler uten å påvirke romlig oppløsning

    def forward(self, x):
        # Forward pass gjennom konvolusjonslag for sluttresultat
        return self.conv(x)

# UNet-arkitektur for bildesegmentering med enkoding og dekoding
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        n_channels: Antall kanaler i input (f.eks. 3 for RGB)
        n_classes: Antall klasser i segmenteringsoutput
        bilinear: Boolsk, om bilineær oppskalering skal brukes i dekoder-delen
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Definerer UNet-blokker
        self.inc = DoubleConv(n_channels, 64)  # Inngangsblokk med 64 kanaler
        self.down1 = Down(64, 128)             # Første nedskaleringsblokk
        self.down2 = Down(128, 256)            # Andre nedskaleringsblokk
        self.down3 = Down(256, 512)            # Tredje nedskaleringsblokk
        factor = 2 if bilinear else 1          # Hvis bilineær oppskalering, reduserer vi kanaler med faktor 2
        self.down4 = Down(512, 1024 // factor) # Fjerde nedskaleringsblokk
        self.up1 = Up(1024, 512 // factor, bilinear) # Første oppskaleringsblokk
        self.up2 = Up(512, 256 // factor, bilinear)  # Andre oppskaleringsblokk
        self.up3 = Up(256, 128 // factor, bilinear)  # Tredje oppskaleringsblokk
        self.up4 = Up(128, 64, bilinear)             # Fjerde oppskaleringsblokk
        self.outc = OutConv(64, n_classes)           # Sluttblokk for output

    def forward(self, x):
        """
        x: Input bilde til segmenteringsmodellen.
        Returnerer segmenteringskartet.
        """
        # Forward pass gjennom UNet-blokkene
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """
        Aktiverer checkpointing for å redusere minnebruk under opplæring.
        Bruk self.use_checkpointing() for å aktivere denne funksjonaliteten.
        """
        # Aktiverer checkpointing på hvert lag i modellen
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
