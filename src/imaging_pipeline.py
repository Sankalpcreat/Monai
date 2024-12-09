import torch
from monia.networks.nets import UNet
from monia.transforms import ( Compose,LoadImage,AddChannel,ScaleIntensity,EnsureType)

class ImagingPipeline:
    def __init__(self,model_path="models/monai_model_zoo/spleen_ct_segmentation/models/model.pt",device="cpu"):
        self.device=device

        self.model=UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16,32,64,128,256),
            strides=(2,2,2,2),
            num_res_units=2
        ).to(device)


        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.eval()

        self.preprocess=Compose([
            LoadImage(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            EnsureType()
        ])
    
    def infer_image(self,image_path:str):
        img=self.preprocess(image_path).to(self.device)

        with torch.no_grad():
            output = self.model(img.unsqueeze(0))
        
        prediction=torch.argmax(output,dim=1).cpu().numpy()[0]

        return prediction


