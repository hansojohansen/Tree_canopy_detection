import numpy as np
import os
import torch
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataloader import segDataset  # Importer segmenteringsdatasettloader
from losses import FocalLoss, mIoULoss  # Importer tapsfunksjoner
from model import UNet  # Importer UNet-modell
from torch import nn
from datetime import datetime
from tqdm import tqdm  # Bibliotek for fremdriftsindikatorer

threshold = 0.5  # Høyere verdi betyr mer konservativ, 0,5 er standard

# Funksjon for å håndtere kommandolinjeargumenter
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Database', help='Rotbane til datasettet')
    parser.add_argument('--num_epochs', type=int, default=2, help='Antall epoker')
    parser.add_argument('--batch', type=int, default=4, help='Batch-størrelse')
    parser.add_argument('--loss', type=str, default='crossentropy', help='Tapsfunksjon som skal brukes: focalloss, iouloss, eller crossentropy')
    return parser.parse_args()

# Definer transformasjoner for datasettet (bare konverterer til tensor her)
t = transforms.Compose([transforms.ToTensor()])

# Funksjon for å beregne nøyaktighet
def acc(label, predicted):
    # Returnerer andelen piksler som er korrekt klassifisert
    return (label.cpu() == torch.argmax(predicted, axis=1).cpu()).sum() / torch.numel(label.cpu())

# Funksjon for å beregne presisjon, tilbakekalling og F1-score
def compute_metrics(label, predicted):
    # Både label og predicted er tensorer med form (batch_size, høyde, bredde)
    # med klasseindekser (0 eller 1 for binær segmentering)
    tp = ((label == 1) & (predicted == 1)).sum().item()
    fp = ((label == 0) & (predicted == 1)).sum().item()
    fn = ((label == 1) & (predicted == 0)).sum().item()
    # Unngå divisjon med null
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def visualize_predictions(model, dataloader, device, num_images=3, save_dir=None):
    # [Uendret visualiseringsfunksjon]
    ...

# Hovedseksjon: oppsett og trening av modellen
if __name__ == '__main__':
    args = get_args()
    train_dataset = segDataset(os.path.join(args.data, 'training'), patch_size=124, transform=t)
    val_dataset = segDataset(os.path.join(args.data, 'validating'), patch_size=124, transform=t)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)

    if args.loss == 'focalloss':
        criterion = FocalLoss(gamma=3/4).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=2).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError("Ustøttet tapsfunksjon")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    best_val_loss = float('inf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'./saved_models/training_{timestamp}/'
    os.makedirs(save_dir, exist_ok=True)

    total_batches = len(train_dataloader) * args.num_epochs

    # Lister for å lagre tap og nøyaktigheter for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Lister for å lagre presisjon, tilbakekalling og F1-score
    train_precisions = []
    train_recalls = []
    train_f1s = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Hoved treningssløyfe med fremdriftsindikatorer for epoke og total trening
    with tqdm(total=total_batches, desc="Total Treningsfremgang", unit="batch") as total_pbar:
        for epoch in range(args.num_epochs):
            model.train()
            loss_list = []
            acc_list = []
            precision_list = []
            recall_list = []
            f1_list = []

            # Fremdriftsindikator for den enkelte epoken
            with tqdm(total=len(train_dataloader), desc=f"Epoke {epoch+1}/{args.num_epochs}", leave=False) as epoch_pbar:
                for batch_i, (x, y, _, _) in enumerate(train_dataloader):
                    pred_annotation = model(x.to(device))
                    loss = criterion(pred_annotation, y.to(device))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Beregn sannsynligheter og bruk terskelverdi
                    probs = torch.softmax(pred_annotation, dim=1)
                    class1_probs = probs[:, 1, :, :]
                    predicted_labels = (class1_probs > threshold).long()

                    # Beregn nøyaktighet
                    acc_value = (y.cpu() == predicted_labels.cpu()).sum() / torch.numel(y.cpu())
                    acc_list.append(acc_value.item())

                    # Beregn presisjon, tilbakekalling, F1-score
                    precision, recall, f1 = compute_metrics(y.cpu(), pred_annotation, threshold=threshold)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

                    # Oppdater fremdriftsindikator for epoken og total trening
                    epoch_pbar.update(1)
                    total_pbar.update(1)

            # Logg tap, nøyaktigheter og metrikker for plotting
            avg_train_loss = np.mean(loss_list)
            avg_train_acc = np.mean(acc_list)
            avg_train_precision = np.mean(precision_list)
            avg_train_recall = np.mean(recall_list)
            avg_train_f1 = np.mean(f1_list)

            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            train_precisions.append(avg_train_precision)
            train_recalls.append(avg_train_recall)
            train_f1s.append(avg_train_f1)

            # Beregning av valideringstap og nøyaktighet
            model.eval()
            val_loss_list = []
            val_acc_list = []
            val_precision_list = []
            val_recall_list = []
            val_f1_list = []
            for x, y, _, _ in val_dataloader:
                with torch.no_grad():
                    pred_annotation = model(x.to(device))
                val_loss = criterion(pred_annotation, y.to(device)).cpu().item()
                val_loss_list.append(val_loss)
                val_acc_list.append(acc(y, pred_annotation).item())

                # Beregn presisjon, tilbakekalling og F1-score
                predicted_labels = torch.argmax(pred_annotation, axis=1)
                precision, recall, f1 = compute_metrics(y.cpu(), predicted_labels.cpu())
                val_precision_list.append(precision)
                val_recall_list.append(recall)
                val_f1_list.append(f1)

            avg_val_loss = np.mean(val_loss_list)
            avg_val_acc = np.mean(val_acc_list)
            avg_val_precision = np.mean(val_precision_list)
            avg_val_recall = np.mean(val_recall_list)
            avg_val_f1 = np.mean(val_f1_list)

            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            val_precisions.append(avg_val_precision)
            val_recalls.append(avg_val_recall)
            val_f1s.append(avg_val_f1)

            # Skriv ut resultater for epoken
            print(f"\nEpoke {epoch+1} - Trenings Tap: {avg_train_loss:.5f} - Validerings Tap: {avg_val_loss:.5f}")
            print(f"Trenings Presisjon: {avg_train_precision:.5f} - Trenings Tilbakekalling: {avg_train_recall:.5f} - Trenings F1: {avg_train_f1:.5f}")
            print(f"Validerings Presisjon: {avg_val_precision:.5f} - Validerings Tilbakekalling: {avg_val_recall:.5f} - Validerings F1: {avg_val_f1:.5f}")

            # Lagre modell-sjekkpunkter
            torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch}.pth')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
                print(f"Beste modell oppdatert og lagret i {save_dir}")

            # Oppdater læringsrate-scheduler om nødvendig
            lr_scheduler.step()

    # Plot og lagre tap og nøyaktighetsgrafer
    def plot_and_save(metric_values, metric_name):
        plt.figure(figsize=(10, 5))
        plt.plot(metric_values['train'], label=f'Trenings {metric_name}')
        plt.plot(metric_values['val'], label=f'Validerings {metric_name}')
        plt.xlabel("Epoke")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} Over Epoker")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/{metric_name.lower()}_plot.png")
        plt.close()

    # Lagre grafer for tap og nøyaktighet
    plot_and_save({'train': train_losses, 'val': val_losses}, "Tap")
    plot_and_save({'train': train_accuracies, 'val': val_accuracies}, "Nøyaktighet")
    plot_and_save({'train': train_precisions, 'val': val_precisions}, "Presisjon")
    plot_and_save({'train': train_recalls, 'val': val_recalls}, "Tilbakekalling")
    plot_and_save({'train': train_f1s, 'val': val_f1s}, "F1 Score")

    # Visualiser prediksjoner på valideringssettet og lagre som PNG
    print("\nVisualiserer resultater på valideringssettet...")
    visualize_predictions(model, val_dataloader, device, num_images=3, save_dir=save_dir)
