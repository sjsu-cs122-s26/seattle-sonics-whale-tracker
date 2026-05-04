from torchvision import transforms

# Matches eval_tf in train_fish_classifier.ipynb — must stay in sync with training
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
