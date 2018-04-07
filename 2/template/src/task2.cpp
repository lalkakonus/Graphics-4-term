#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "matrix.h"
#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

Matrix <float> brightness(BMP * image);
Matrix <pair <float, float>> gradient(Matrix <float> & brigtness);
Matrix <pair <float, float>> abs_and_angel(Matrix <pair <float, float>> grad);
vector <float> gist(int N, int M, int S0, Matrix <pair <float, float>> data);

// Exatract features from dataset.
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
	BMP * image;
	int target;
    
	for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx){
		image = data_set[image_idx].first;
		target = data_set[image_idx].second;

		Matrix <float> Y = brightness(image);
		Matrix <pair <float, float>> grad = gradient(Y);
		Matrix <pair <float, float>> dir_abs = abs_and_angel(grad);        
        features -> push_back(make_pair(gist(10, 10, 10, dir_abs), target));
     };
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

unsigned max(unsigned a, unsigned b){
	if (a > b)
		return a;
	else
		return b;
}

Matrix <float> custom(Matrix <float> src_matrix, Matrix<float> kernel) {	
	float  sum = 0;
	unsigned k_cols = kernel.n_cols;
	unsigned k_rows = kernel.n_rows;
	
	int offset = max(k_rows / 2 , k_cols / 2);

	Matrix <float> result = src_matrix.extra_borders(offset, offset);
	Matrix <float> tmp = src_matrix.extra_borders(offset, offset);

	//Apply kernel for each item
	for (unsigned x = 0; x < src_matrix.n_cols; ++x)
		for (unsigned y = 0; y < src_matrix.n_rows; ++y)
		{
			//Set sum of each kernel to zero
			sum = 0; 

			//Calculate convolution
			for (unsigned i = 0; i < k_cols; ++i)
				for (unsigned j = 0; j < k_rows; ++j)
					sum += kernel(j, i) * tmp(y + j, x + i);
			//Rewrite result to item
			result(y + (k_rows / 2), x +  (k_cols / 2)) = sum; 
	}

	//Return image with original size
	return result.submatrix(offset, offset, src_matrix.n_rows, src_matrix.n_cols);
}

Matrix <float> brightness(BMP * image){
	int heigth = image -> TellHeight();
	int width = image -> TellWidth();

	Matrix<float> brigtness(heigth, width);
	RGBApixel pixel;

	for (int y = 0; y < heigth; ++y)
		for (int x = 0; x < width; ++x)
		{
			pixel = image -> GetPixel(x, y);
			brigtness(y, x) = 0.299 * pixel.Red + 0.587 * pixel.Green + 0.114 * pixel.Blue;
		}
	return brigtness;
}

Matrix <pair <float, float>> gradient(Matrix <float> & brigtness){
	Matrix <float> x_sobel = {{0,  0, 0},
							  {-1, 0, 1},
							  {0,  0, 0}};

	Matrix <float> y_sobel = {{0,  1, 0},
							  {0,  0, 0},
						      {0, -1, 0}};

	Matrix <float> x_grad = custom(brigtness, x_sobel);	
	Matrix <float> y_grad = custom(brigtness, y_sobel);

	Matrix <pair <float, float>> grad(brigtness.n_rows, brigtness.n_cols);

	for (unsigned i = 0; i < brigtness.n_rows; i++)
		for (unsigned j = 0; j < brigtness.n_cols; j++)
			grad(i, j) = make_pair(x_grad(i, j), y_grad(i, j));
	
	return grad;
}

Matrix <pair <float, float>> abs_and_angel(Matrix <pair <float, float>> grad){
	Matrix <pair <float, float>> result(grad.n_rows, grad.n_cols);	
	float x, y;
	for (unsigned i = 0; i < result.n_rows; i++)
		for (unsigned j = 0; j <	result.n_cols; j++){
			x = grad(i,j).first;
			y = grad(i,j).second;
			result(i,j) = make_pair(sqrt(x * x + y * y), atan2(y, x));
		};
	return result;
}

#define PI 3.14159265
vector<float> gist(int N, int M, int S, Matrix <pair <float, float>> data){
	vector <float> feature_vect;
	Matrix <float> tmp(N*M, S);

	for (int i = 0; i < N*M; i++)
		for (int j = 0; j < S; j++)
			tmp(i, j) = 0;

	int x_size = float(data.n_cols) / M - 1;
	int y_size = float(data.n_rows) / N - 1;
	int dir_class = 0;
	float x_cell, y_cell;

	for (unsigned i = 0; i < data.n_rows; i++)
		for (unsigned j = 0; j < data.n_cols; j++){
			dir_class = data(i, j).second / (PI / S) + S/2;
			x_cell = j / x_size;
			y_cell = i / y_size;
			tmp(x_cell + y_cell * M, dir_class) = tmp(x_cell + y_cell * M, dir_class) + data(i, j).first;
		};

	for (int i = 0; i < N*M; i++)
		for (int j = 0; j < S; j++)
			feature_vect.push_back(tmp(i, j));
	
	return feature_vect;
}
	
// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
