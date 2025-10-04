# AISC-Spring-Project

ML model to detect various types of brain tumors (or no brain tumors). Part of AISC Beginners' Project Cohort Spring 2025.
## About the Model
### Model Description
* ML model to detect various types of brain tumors (or no brain tumors).
* Dataset: ~7,000 RGB brain scans.
* Algorithm used: Convolutional Neural Network (CNN)
* Dataset split 80-20 percent for testing and training respectively
* 30 epochs
  * Stops testing after no improvement after 7 epochs
### Outcomes
* Training accuracy (30th epoch): 98.78%; Testing accuracy: 98.93%
* Classification report:
  * <img width="551" height="205" alt="Classification_Report" src="https://github.com/user-attachments/assets/56c65f31-3240-4690-b7d9-8b980556f2b8" />
* Confusion Matrix:
  * <img width="677" height="549" alt="Confusion_Matrix" src="https://github.com/user-attachments/assets/74070862-af03-4747-b756-1b3472752c07" />
## Dependencies
* **Core AI/ML:** PyTorch, Scikit-learn
* **Data handling:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Image processing:** Pillow
* **Utilities:** OS, Random
## Credits
Jonathan Cruz (Project Manager), Shaffana Mustafa, Keren Skariah, Matthew Wang, Saanika Gupta, Sathvik Parasa, Sreeya Yakkala, Wang Nguyen
