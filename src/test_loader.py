from dataset import train_loader, validation_loader, test_loader


# Get one training batch
images, labels = next(iter(train_loader))

print("Training batch")
print("-------------------------")
print("Image shape :", images.shape)
print("Label shape :", labels.shape)
print("Labels      :", labels)
print()


# Check validation batch
images, labels = next(iter(validation_loader))

print("Validation batch")
print("-------------------------")
print("Image shape :", images.shape)
print("Label shape :", labels.shape)
print()


# Check test batch
images, labels = next(iter(test_loader))

print("Testing batch")
print("-------------------------")
print("Image shape :", images.shape)
print("Label shape :", labels.shape)