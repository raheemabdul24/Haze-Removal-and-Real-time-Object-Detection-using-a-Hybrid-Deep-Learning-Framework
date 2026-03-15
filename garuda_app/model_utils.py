import torch
import torch.nn as nn
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SWDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_model():

    model = SWDModel().to(device)

    model.load_state_dict(
        torch.load("models/dehazing_best_model.pth", map_location=device)
    )

    model.eval()

    return model


def enhance_image(model, frame):

    original = frame.copy()

    img = cv2.resize(frame,(256,256))
    img = img/255.0
    img = np.transpose(img,(2,0,1))

    tensor = torch.tensor(img,dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().cpu().numpy()
    output = np.transpose(output,(1,2,0))
    output = (output*255).astype(np.uint8)

    output = cv2.resize(output,(original.shape[1], original.shape[0]))

    # Blend original + enhanced
    output = cv2.addWeighted(original,0.6,output,0.4,0)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    output = cv2.filter2D(output,-1,kernel)

    # CLAHE contrast
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    output = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

    return original, output
