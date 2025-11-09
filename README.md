# Project5_FishClassification-through-Deep-Learning
What the Project Does?
This project focuses on building a deep learning-based fish species classification system.  It uses advanced Convolutional Neural Networks (CNNs) and transfer learning techniques to accurately classify different types of fish from images.
The final deployed version of the model allows users to:
- Upload a fish image through a Streamlit web app.
- Select a pre-trained model (VGG16, ResNet50, InceptionV3, MobileNetV2, EfficientNetB0).
- Get the predicted fish species name and model confidence score.
- Compare top-5 prediction probabilities and view a model performance summary side-by-side.

Dataset Description:
The dataset used contains **11 fish species**, each categorized into subfolders under the directories: train, val,test.
Each folder contains images of:
- `animal_fish`
- `animal_fish_bass`
- `fish_sea_food_black_sea_sprat`
- `fish_sea_food_gilt_head_bream`
- `fish_sea_food_hourse_mackerel`
- `fish_sea_food_red_mullet`
- `fish_sea_food_red_sea_bream`
- `fish_sea_food_sea_bass`
- `fish_sea_food_shrimp`
- `fish_sea_food_striped_red_mullet`
- `fish_sea_food_trout`

Technologies Used:
Language: Python 3.10+  
Deep Learning Framework: TensorFlow / Keras  
Data Handling: NumPy, Pandas  
Model Evaluation: scikit-learn  
Visualization: Matplotlib, Seaborn  
Deployment: Streamlit  
Model Saving: .keras and .h5 formats  
Other Utilities: tqdm, Pillow, python-dotenv 

 Why the Project Is Useful?
- Automates fish species identification using deep learning, applicable in marine research, seafood processing, and fishery automation.  
- Provides transparent performance comparison between multiple CNN architectures.  
- Integrates a user-friendly web interface for non-technical users.  
- Demonstrates end-to-end machine learning workflow:  
  Data preprocessing → Model training → Fine-tuning → Evaluation → Deployment.

 Where Users Can Get Help?
 For specific questions, open a new issue with detailed descriptions.  You can reach out via email at: geethabalan96@gmail.com, Kindly check Fish Image Clustering.pptx for reference.

Who Maintains and Contributes to the Project?
Author: Geetha Palanisamy  
Project Type: Academic Deep Learning Project  
Guidance: Developed with practical experimentation in model evaluation, fine-tuning, and deployment using TensorFlow and Streamlit.  

Contributions, suggestions, and pull requests are welcome — please open an issue before making major changes.

