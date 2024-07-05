import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.data_utils import load_config, filter_and_prepare_tensor, normalize_tensor, create_sequences
from data.dataloader import create_dataloader
from models.lstm_model import LSTMModel 
from utils.data_processing import compute_correlation_matrix, plot_correlation_heatmap


def main():
    # Carica la configurazione
    config = load_config('config/config.json', 'config/config_schema.json')

    # Carica il dataset
    file_path = config['data']['file_path']
    df = pd.read_csv(file_path)

    # Filtraggio e preparazione dei tensori
    filters_pil = {"tipo_dato": ["B1GQ_B_W2_S1_X2"], "edizione": ["Set-2023"]}
    filters_prod = {"tipo_dato": ["P1_C_W2_S1"], "edizione": ["Set-2023"], "valutazione": ["prezzi correnti"]}
    filters_rlg = {"tipo_dato": ["P3_D_W0_S1"], "edizione": ["Set-2023"], "valutazione": ["prezzi correnti"]}
    filters_rm = {"tipo_dato": ["B2A3G_B_W2_S1"], "edizione": ["Set-2023"]}

    columns_to_remove = ['Seleziona periodo', 'Flag Codes', 'Flags']

    tensor_pil = filter_and_prepare_tensor(df, filters_pil, columns_to_remove)
    tensor_prod = filter_and_prepare_tensor(df, filters_prod, columns_to_remove)
    tensor_rlg = filter_and_prepare_tensor(df, filters_rlg, columns_to_remove)
    tensor_rm = filter_and_prepare_tensor(df, filters_rm, columns_to_remove)

    tensor_pil_normalized = normalize_tensor(tensor_pil)
    tensor_prod_normalized = normalize_tensor(tensor_prod)
    tensor_rlg_normalized = normalize_tensor(tensor_rlg)
    tensor_rm_normalized = normalize_tensor(tensor_rm)

    # Creazione delle sequenze e dei target
    seq_length = config['data']['seq_length']
    sequences, targets = create_sequences(tensor_pil_normalized, tensor_prod_normalized, tensor_rlg_normalized, tensor_rm_normalized, seq_length=seq_length)

    # Creazione del DataLoader
    batch_size = config['training']['batch_size']
    dataloader = create_dataloader(sequences, targets, batch_size)

    # Parametri del modello
    input_size_pil = tensor_pil_normalized.shape[1]
    input_size_prod = tensor_prod_normalized.shape[1]
    input_size_rlg = tensor_rlg_normalized.shape[1]
    input_size_rm = tensor_rm_normalized.shape[1]
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    output_size = config['model']['output_size']

    # Creazione del modello
    model = LSTMModel(input_size_pil, input_size_prod, input_size_rlg, input_size_rm, hidden_size, num_layers, output_size)

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Addestramento del modello
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_seq_pil, batch_seq_prod, batch_seq_rlg, batch_seq_rm, batch_target in dataloader:
            optimizer.zero_grad()
            output = model(batch_seq_pil, batch_seq_prod, batch_seq_rlg, batch_seq_rm)
            loss = criterion(output.squeeze(), batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

    # Previsioni future
    num_predictions = config['prediction']['num_predictions']
    predicted_values = []
    model.eval()
    with torch.no_grad():
        last_sequence_pil = sequences[-1][0].unsqueeze(0)  # Ultima sequenza PIL
        last_sequence_prod = sequences[-1][1].unsqueeze(0)  # Ultima sequenza Produzione
        last_sequence_rlg = sequences[-1][2].unsqueeze(0)  # Ultima sequenza RLG
        last_sequence_rm = sequences[-1][3].unsqueeze(0)  # Ultima sequenza RM
        for _ in range(num_predictions):
            predicted_value_normalized = model(last_sequence_pil, last_sequence_prod, last_sequence_rlg, last_sequence_rm).item()
            predicted_value = predicted_value_normalized * (tensor_pil.max(dim=0, keepdim=True).values[0, 1] - tensor_pil.min(dim=0, keepdim=True).values[0, 1]) + tensor_pil.min(dim=0, keepdim=True).values[0, 1]
            predicted_values.append(predicted_value)

            # Aggiorna le sequenze con la nuova previsione
            new_sequence_pil = torch.cat((last_sequence_pil[:, 1:, :], torch.tensor([[[tensor_pil[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)
            new_sequence_prod = torch.cat((last_sequence_prod[:, 1:, :], torch.tensor([[[tensor_prod[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)
            new_sequence_rlg = torch.cat((last_sequence_rlg[:, 1:, :], torch.tensor([[[tensor_rlg[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)
            new_sequence_rm = torch.cat((last_sequence_rm[:, 1:, :], torch.tensor([[[tensor_rm[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)
            
            last_sequence_pil = new_sequence_pil
            last_sequence_prod = new_sequence_prod
            last_sequence_rlg = new_sequence_rlg
            last_sequence_rm = new_sequence_rm

    # Stampa le previsioni
    for i, value in enumerate(predicted_values, 1):
        print(f'Predicted value for year {int(tensor_pil[-1, 0].item()) + i}: {value:.2f}')

    # Calcola e stampa la matrice di correlazione
    correlation_matrix, merged_df = compute_correlation_matrix(df)
    print("Matrice di correlazione:")
    print(correlation_matrix)

if __name__ == "__main__":
    main()

