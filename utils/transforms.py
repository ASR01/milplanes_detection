import albumentations as A

from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms

# Training Transformation
def get_train_aug():
    return A.Compose([
        A.MotionBlur(blur_limit=3, p=0.33),
        A.Blur(blur_limit=3, p=0.33),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, p=0.33
        ),
        A.ColorJitter(p=0.33),
        A.Rotate(limit=90, p=0.2),
        A.RandomGamma(p=0.2),
        A.RandomFog(p=0.2),
        A.RandomSunFlare(p=0.1),
        A.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# Define the transform for train images

def get_train_transform():
    return A.Compose([
        A.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# Define the transform for validation imagess
def get_val_transform():
    return A.Compose([
        A.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
        ToTensorV2(p=1.0),
        ], bbox_params={ 
                    'format': 'pascal_voc', 
                    'label_fields': ['labels']
                    }
    )

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)