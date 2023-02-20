# modelling-airbnbs-property-listing-dataset-

A set of images and tabular data was downloaded to be used for this project from Airbnb. 
Firstly, the images were resized such that the height of every image was the same as the height of the smallest image in that particular folder.
Images not in RGB format were also removed and the new resized images were saved into their own new folders in a directory labelled as 'processed images'
This function was then stored in a if __name__ == '__main__' block to only run if its the main file and not an imported file.

Next, the tabular data was analysed and cleaned by firstly removing the empty rows containing no ratings. Next, the description column was turned into a list from a previous list of strings using the literal_eval method imported from ast.
For some of the numerical columns containing NA values, they were automatically switched to a default value of 1.
The code for cleaning this tabular data was stored in a clean_tabular_data method that runs in an if __name__ == '__main__' block.
A load_airbnb data method was created using only the numerical tabular data where a particular column is used as a label and the corresponding features (the remaining tabular data excluding the label) are also returned. 
This returns a tuple containing the label and features once called.
