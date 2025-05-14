import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# FocalLoss-klassen brukes til å håndtere ubalanser mellom klasser ved å fokusere mer på vanskeligere eksempler.
# Dette er spesielt nyttig i segmentering med ubalanserte data (dvs. hvor enkelte klasser forekommer sjeldent).
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        gamma: Kontrollerer fokuseringen på vanskeligere eksempler. Høyere verdi betyr større fokus på feilklassifiserte eksempler.
        alpha: Vekt for klassene, nyttig når en klasse forekommer betydelig sjeldnere enn andre.
        size_average: Hvis True, returnerer gjennomsnittlig tap (loss) over batchen, ellers summen.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    # Metoden `forward` behandler hvert piksel separat i batchen.
    def forward(self, input, target):
        # For segmentering: flater ut alle dimensjoner for enkel behandling
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # Endrer N, C, H, W => N, C, H*W
            input = input.transpose(1, 2)    # Endrer N, C, H*W => N, H*W, C
            input = input.contiguous().view(-1, input.size(2))   # N, H*W, C => N*H*W, C
        target = target.view(-1, 1)

        # `logpt` er log-sannsynligheten for klassen med "True" label per piksel
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())  # Sannsynlighet for "True" klassen

        # Hvis alpha er spesifisert, brukes den til å vekte klassene ulikt
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        # Focal loss-formel: høyere gamma-verdier reduserer tapet for riktig klassifiserte eksempler
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()


# Klassen `mIoULoss` beregner "mean intersection over union loss", en vanlig målemetode i segmentering.
# Lossen beregnes som `1 - IoU` der lavere verdi indikerer bedre resultat.
class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        """
        weight: Vekt for hver klasse i beregning av loss, ofte brukt ved ubalanserte klasser.
        size_average: Hvis True, returnerer gjennomsnittlig tap (loss), ellers summen.
        n_classes: Antall klasser i segmenteringen (default: 2).
        """
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    # Konverterer et tensor til "one-hot" representasjon for beregning av IoU per klasse
    def to_one_hot(self, tensor):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, self.classes, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    # Beregner IoU loss ved å sammenligne modellens prediksjoner og sannhetsannotatione per klasse.
    def forward(self, inputs, target):
        # `inputs`: modellens output som sannsynligheter per klasse per piksel
        # `target_oneHot`: sannhetsverdier som "one-hot" for hver klasse
        N = inputs.size()[0]  # Antall i batch

        # Beregner prediksjonssannsynligheter per piksel
        inputs = F.softmax(inputs, dim=1)
        
        # Multipliserer prediksjonene med "one-hot" for å kalkulere overlapp (krysning)
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        # Summerer over alle piksler, N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Beregner unionen: inputs + target_oneHot - overlap, og summerer over piksler
        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)

        # Beregner IoU loss ved å ta forholdet mellom overlapp og union, deretter 1 - IoU
        loss = inter / union
        return 1 - loss.mean()  # Returnerer gjennomsnittlig tap over klasser og batch
