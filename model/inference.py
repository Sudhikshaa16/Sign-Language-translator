# import torch
# from model.model_lstm import LSTMModel
# import numpy as np

# GESTURES = ["HELLO","YES","NO","PLEASE","THANKYOU","SORRY","HELP","STOP","ILOVEYOU","GOODBYE"]

# class GestureRecognizer:
#     def __init__(self, model_path, device="cpu"):
#         self.device = device
#         self.gestures = GESTURES  # define gestures here
#         self.model = LSTMModel(input_size=42, hidden_size=128, output_size=len(GESTURES))
#         self.model.load_state_dict(torch.load(model_path, map_location=device))
#         self.model.eval()
#         self.model.to(self.device)
    
#     def predict(self, landmarks):
#         """
#         landmarks: np.array of shape (21, 2) for a single frame,
#                    or (seq_len, 42) if already flattened sequence
#         """
#         x = torch.tensor(landmarks, dtype=torch.float32, device=self.device)

#         # Ensure shape = (batch, seq_len, input_size)
#         if len(x.shape) == 2:  # single frame (21,2) -> flatten to (1, 1, 42)
#             x = x.view(1, 1, -1)
#         elif len(x.shape) == 1:  # single flattened frame (42,)
#             x = x.view(1, 1, -1)
#         elif len(x.shape) == 3:
#             x = x.unsqueeze(0)  # add batch dimension if missing

#         out = self.model(x)  # shape: (batch, output_size)
#         pred = torch.argmax(out, dim=1).item()
#         return self.gestures[pred]


import torch
from model.model_lstm import LSTMModel
import numpy as np

# IMPORTANT: Use only the gestures you actually have
GESTURES = ["HELLO", "YES", "NO", "PLEASE", "THANKYOU"]

class GestureRecognizer:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.gestures = GESTURES
        
        # Your LSTM model must match your training
        self.model = LSTMModel(
            input_size=42,       # 21 landmarks * 2 coords
            hidden_size=128, 
            output_size=len(GESTURES)
        )
        
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.eval()
        self.model.to(self.device)

    def predict(self, landmarks):
        """
        landmarks can be:
            (21, 2)  → one frame
            (42,)    → flattened frame
            (seq, 42) → sequence of frames
        """
        x = torch.tensor(landmarks, dtype=torch.float32, device=self.device)

        # --- Shape Normalization ---
        if len(x.shape) == 2:            # (21,2)
            x = x.flatten().view(1, 1, -1)

        elif len(x.shape) == 1:          # (42,)
            x = x.view(1, 1, -1)

        elif len(x.shape) == 2 and x.shape[1] == 42:  # (seq, 42)
            x = x.unsqueeze(0)

        out = self.model(x)
        pred = torch.argmax(out, dim=1).item()
        return self.gestures[pred]
