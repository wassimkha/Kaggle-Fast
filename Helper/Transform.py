import torchvision.transforms as transform

#Transformations
Norm = transform.Compose([transform.Resize((224,224)),transform.ToTensor(), transform.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
To_tensor = transform.ToTensor()
Resize = transform.Resize((224,224))
To_pil = transform.ToPILImage()
