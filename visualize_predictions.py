import torch.utils.data
from utils.args import parse_args
from utils.utilities import get_test_loaders
import os
from tqdm import tqdm
import cv2

# Visualize the model predictions
if __name__ == '__main__':
    if not os.path.exists('./output_img'):
        os.mkdir('./output_img')

    parser, metadata = parse_args()
    opt = parser.parse_args()

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_loader = get_test_loaders(opt, batch_size=1) # Get the test loader
    test_img_paths = [image for image in os.listdir(opt.dataset_dir + 'test/A/') if not
                    image.startswith('.')] # Get the test image paths (Used for saving the output images)
    test_img_paths.sort() # Sort the image paths

    path = './checkpoints/checkpoint_epoch_17.pt'   # the path of the model
    model = torch.load(path) # Load the model
    
    model.eval()
    index_img = 0 # Used to index images for saving the output images
    with torch.no_grad():
        tbar = tqdm(test_loader)
        for batch_img1, batch_img2, labels in tbar:
            # Convert the images to float and move them to the device
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            preds = model(batch_img1, batch_img2) # Get the predictions from the model

            preds = preds[-1] 
            _, preds = torch.max(preds, 1)
            preds = preds.data.cpu().numpy() # Convert the predictions to numpy array
            preds = preds.squeeze() * 255  # Convert the predictions to 0-255 range (image)

            # Save the output image
            file_path = './output_img/' + str(test_img_paths[index_img]).zfill(4)
            cv2.imwrite(file_path, preds)

            index_img += 1
