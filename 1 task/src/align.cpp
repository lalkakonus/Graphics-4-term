//Kononov Sergey 311 group

#include "align.h"
#include <string>
#include <cmath>
#include <vector>

using std::string;
using std::cout;
using std::endl;

using std::sort;
using std::get;
using std::tie;
using std::vector;
using std::tuple;
using std::make_tuple;

#define MAX_X_OFF  20
#define MAX_Y_OFF  20

void update(Matrix <unsigned> tmp, vector <tuple <unsigned, unsigned>> add, bool value);

bool out_border(int a, int b, int N, int M);

bool examine_surround(int y, int x, Matrix <bool> check, Matrix <unsigned> tmp, vector <tuple <unsigned, unsigned>> array);

int border(int * array, int size, bool flag);

int make_sum(const Image & Image1, const Image & Image2, const int x_of, const int y_of)
{
	int sum = 0, tmp_x = 0, tmp_y = 0;

	for (unsigned j = 0; j < Image1.n_cols; ++j)
		for (unsigned i = 0; i < Image1.n_rows; ++i)
		{
			tmp_x = j + x_of;
			tmp_y = i + y_of;
			//Control that we inside pictire
			if ((tmp_x < 0) || (tmp_x > int(Image1.n_cols) - 1) || (tmp_y < 0) || (tmp_y > int(Image1.n_rows) - 1))
				continue;
			sum += pow(int(get<1>(Image1(i, j))) - int(get<1>(Image2(tmp_y, tmp_x))) , 2);
		}

	//Norm sum
	sum = sum / int((Image1.n_cols - abs(x_of))*(Image1.n_rows - abs(y_of)));
	
	return sum;
}

Image connect_two(const Image Image1, const Image Image2, int flag = 0)
{
	int x_offset = 0, y_offset = 0, current_sum = 0, tmp = 0, tmp_x = 0, tmp_y = 0;
	
	Image result = Image1.deep_copy();

	//Calculate initial sum with zero offset
	current_sum = make_sum(Image1, Image2, 0, 0);

	//Calculte sum for each offset
	for (int x = - MAX_X_OFF; x < MAX_X_OFF + 1; ++x)
		for (int y = - MAX_Y_OFF; y < MAX_Y_OFF + 1; ++y)
		{
			//find minimal sum
			tmp = make_sum(Image1, Image2, x, y);
			
			//save offset and value
			if (tmp < current_sum)
			{
				x_offset = x;
				y_offset = y;
				current_sum = tmp;
			}
		}

	//flag 0 mean that we change B color, 1 that R color
	for (unsigned i = 0; i < Image1.n_cols; ++i)
		for (unsigned j = 0; j < Image1.n_rows; ++j)
		{
			tmp_x = i + x_offset;
			tmp_y = j + y_offset;
			if ((tmp_x < 0) || (tmp_x > int(Image1.n_cols) - 1) || (tmp_y < 0) || (tmp_y > int(Image1.n_rows) - 1))
				continue;
			if (flag)
				get<2>(result(j,i)) = get<2>(Image2(tmp_y,tmp_x));
			else
				get<0>(result(j,i)) = get<0>(Image2(tmp_y,tmp_x));

		}

	return result;
}

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale)
{
	int n_rows = srcImage.n_rows, n_cols = srcImage.n_cols;
 
	//Crop image to three layers
	Image imB = srcImage.submatrix(0, 0, n_rows / 3, n_cols);
	Image imG = srcImage.submatrix(n_rows / 3, 0, n_rows / 3, n_cols);
	Image imR = srcImage.submatrix((n_rows / 3) * 2, 0, n_rows / 3, n_cols);
	
	//Connect red and green layer 
	Image imRG = connect_two(imG, imR, 0);
	
	//Connect all layers
	Image imRGB = connect_two(imRG, imB, 2);
	
	return imRGB;
}

Image sobel_x(Image src_image) 
{
	Matrix<double> kernel = {{-1, 0, 1},
                             	 {-2, 0, 2},
                             	 {-1, 0, 1}};
 
	return custom(src_image, kernel);
}

Image sobel_y(Image src_image)
{
	Matrix<double> kernel = {{ 1,  2,  1},
                                 { 0,  0,  0},
                                 {-1, -2, -1}};
 
	return custom(src_image, kernel);
}

Image sobel_x1(Image src_image) 
{
	Matrix<double> kernel = {{1, 0, -1},
                             	 {2, 0, -2},
                             	 {1, 0, -1}};
 
	return custom(src_image, kernel);
}

Image sobel_y1(Image src_image)
{
	Matrix<double> kernel = {{ -1,  -2,  -1},
                                 { 0,  0,  0},
                                 {1, 2, 1}};
 
	return custom(src_image, kernel);
}
Image unsharp(Image src_image)
{
	Matrix <double> kernel = {{ -1/6, -2/3, -1/6},
				  { -2/3, -4/3, -2/3},
				  { -1/6, -2/3, -1/6}};

	return custom(src_image, kernel);
}

Image gray_world(Image src_image)
{
    	double S = 0, Sr = 0, Sg = 0, Sb = 0;
	double Cg = 0, Cb = 0, Cr = 0;

	Image result = src_image.deep_copy();
	
	//Calculate sum for each channel
	for (unsigned y = 0; y < result.n_rows; ++y)
		for (unsigned x =0; x < result.n_cols; ++x)
		{
			Sr += get<0>(result(y, x));
			Sg += get<1>(result(y, x));
			Sb += get<2>(result(y, x));
		}

	//Calculate average value for each channel
	Sr /= result.n_rows * src_image.n_cols;
	Sg /= result.n_rows * src_image.n_cols;
	Sb /= result.n_rows * src_image.n_cols;

	//Caluclate average value
	S = (Sr + Sg + Sb) / 3;

	//Control that average value more than zero
	if (Sr > 0.001) Cr = S / Sr;
	if (Sg > 0.001) Cg = S / Sg;
	if (Sb > 0.001) Cb = S / Sb;

	for (unsigned y = 0; y < src_image.n_rows; ++y)
		for (unsigned x =0; x < src_image.n_cols; ++x)
		{
			get<0>(result(y, x)) *= Cr;
			get<1>(result(y, x)) *= Cg;
			get<2>(result(y, x)) *= Cb;
		
			//Control that each value less than 256
			if (get<0>(result(y, x)) > 255) get<0>(result(y, x)) = 255;
			if (get<1>(result(y, x)) > 255) get<1>(result(y, x)) = 255;
			if (get<2>(result(y, x)) > 255) get<2>(result(y, x)) = 255;
		}

	return result;
}

Image resize(Image src_image, double scale) {
    return src_image;
}

Image custom(Image src_image, Matrix<double> kernel) 
{	
	float  r_sum = 0, b_sum = 0, g_sum = 0, r, g, b;
	const unsigned k_cols = kernel.n_cols;
	const unsigned k_rows = kernel.n_rows;
	int offset = max(k_rows / 2 , k_cols / 2);

	//Expand image with mirroring border
	Image result = mirroring(src_image, offset);
	Image tmp = mirroring(src_image, offset);

	//Apply kernel for each item
	for (unsigned x = 0; x < src_image.n_cols; ++x)
		for (unsigned y = 0; y < src_image.n_rows; ++y)
		{
			//Set sum of each kernel to zerp
			r_sum = g_sum = b_sum = 0; 
			
			//Calculate convolution
			for (unsigned i = 0; i < k_cols; ++i)
				for (unsigned j = 0; j < k_rows; ++j)
				{
					tie(r, g, b) = tmp(y + j, x + i);
					r_sum += kernel(j, i) * r;
					g_sum += kernel(j, i) * g;
					b_sum += kernel(j, i) * b;
				}
			//Rewrite result to item
			result(y + (k_rows / 2), x +  (k_cols / 2)) = make_tuple(r_sum, g_sum, b_sum);
		}
	
	//Return image with original size
	return result.submatrix(offset, offset, src_image.n_rows, src_image.n_cols);
}

Image autocontrast(Image src_image, double fraction) 
{    	
	unsigned gist [256], R = 0, G = 0, B = 0, Y = 0;
	int tmp = 0, min = 0, max = 255;
	const double param = src_image.n_cols * src_image.n_rows * fraction;
	//double coeff = 255.0;
	Image result(src_image.n_rows, src_image.n_cols);


	//Set gistogram array to zero
	for (int i = 0; i < 256; ++i)
		gist[i] = 0; 

	//Fill gistogram array
	for (unsigned x = 0; x < src_image.n_cols; ++x)
		for (unsigned y = 0; y < src_image.n_rows; ++y)
		{
			tie(R, G, B) = src_image(y, x);
			//Y is brightness
			Y = 0.2125 * R + 0.7154 * G + 0.0721 * B;
			gist[Y]++;
		};

	for (int i = 0; i < 256; ++i)
		if ((tmp += gist[i]) > param)
		{
			min = i;
			std::cout << "!";
			break;
		}
	
	tmp = 0;
	for (int i = 0; i < 256; ++i)
		if ((tmp += gist[255 -i]) > param)
		{
			max = 255 - i;
			break;
		}

	for (unsigned x = 0; x < src_image.n_cols; ++x)
		for (unsigned y = 0; y < src_image.n_rows; ++y)
		{
			tie(R, G, B) = src_image(y, x);
			Y = 0.2125 * R + 0.7154 * G + 0.0721 * B;
			if (Y < abs(min))
			{
				result(y, x) = make_tuple(0, 0, 0);
				continue;
			}
			if (Y > abs(max))
			{
				result(y, x) = make_tuple(255, 255, 255);
				continue;
			}

			//Resize values
			int r, g, b;
			r = 250.0 * (R - min) / (max - min);
			g = 255.0 * (G - min) / (max - min);
			b = 255.0 * (B - min) / (max - min);
			
			//Control results
			if (r > 255) r = 255; 
			if (g > 255) g = 255; 
			if (b > 255) b = 255; 

			if (r < 0) r = 0; 
			if (g < 0) g = 0; 
			if (b < 0) b = 0;

			result(y, x) = make_tuple(r, g, b);
		}

	return result;
}

Image gaussian(Image src_image, double sigma, int radius)  {
	//Linear size of kernel
	int N = radius * 2 + 1; 

	Matrix <double> kernel(N, N); 
	float  norm = 0; 
	
	//Calculate gaussian kernael elements
	for (int i = 0; i < N; ++i)
		for (int j = 0; j< N; ++j)
		{
			kernel(j ,i) = (1/ (2 * M_PI * pow(sigma, 2)));
			kernel(j, i) *= exp(-(pow(i - radius, 2) + pow(j - radius, 2)) /(2 * pow(sigma, 2)));
			norm += kernel(j, i);
		};

	//Norm kernel
	for (int i = 0; i < N; ++i)
		for (int j = 0; j< N; ++j)
			kernel(j, i) /= norm;

	//Apply convolution
	return custom(src_image, kernel);
}

Image gaussian_separable(Image src_image, double sigma, int radius)
{
	double  norm = 0;

	//Expand image
	Image result = mirroring(src_image, radius);

	Matrix <double> kernel_y(radius * 2 + 1, 1); 
	Matrix <double> kernel_x(1, radius * 2 + 1); 

	if (radius <= 0) throw "Radius should be more than zero!";

	for (int i = 0; i < radius * 2 + 1; ++i)
	{
		kernel_x(0, i) = (1/ (pow(2 * M_PI, 1/2) * sigma)) * exp(-(pow(i - radius, 2)) /(2 * pow( sigma, 2)));
		norm += kernel_x(0, i);
	};

	// X gaussian kernel is like Y kernel, but tranposed
	for (int i = 0; i < radius * 2 + 1; ++i)
	{
		kernel_x(0, i) /= norm;
		kernel_y(i, 0) = kernel_x(0, i);
	};

	//Apply convolution with X kernel and Y kernel
	result = custom(src_image, kernel_x);
	result = custom(result, kernel_y);

	return result;
}

void surround_median(Image result, Image src_image, int radius, unsigned y, unsigned x)
{
	//vector for each layer
	vector <unsigned> R_array, G_array, B_array;
	
	R_array.clear();
	G_array.clear();
	B_array.clear();
	
	//sum surrounding values of each layer
	for (int i = -radius; i <= radius; ++i)
		for (int j = -radius; j <= radius; ++j)
		{
			R_array.push_back(get<0>(src_image(y + j, x + i)));
			G_array.push_back(get<1>(src_image(y + j, x + i)));
			B_array.push_back(get<2>(src_image(y + j, x + i)));
		}

	//sort values for each layer
	sort(R_array.begin(), R_array.end());
	sort(G_array.begin(), G_array.end());
	sort(B_array.begin(), B_array.end());

	//rewrite values
	get<0>(result(y - radius, x - radius)) = R_array[pow(radius * 2 + 1,2) / 2 ];
	get<1>(result(y - radius, x - radius)) = G_array[pow(radius * 2 + 1,2) / 2 ];
	get<2>(result(y - radius, x - radius)) = B_array[pow(radius * 2 + 1,2) / 2 ];
}

Image median(Image src_image, int radius)
{
	//expand image by mirroring parts near border
	Image result(src_image.n_rows, src_image.n_cols);
	Image expanded_image = mirroring(src_image, radius);

	//calculate median value for each pixel
	for (unsigned x = 0; x < src_image.n_cols; ++x)
		for (unsigned y = 0; y < src_image.n_rows; ++y)
			surround_median(result, expanded_image, radius, y + radius, x + radius);

	//return image with initial size
	return result;
}

Image mirroring(Image src_image, unsigned radius)
{
	Image result(src_image.n_rows + radius * 2, src_image.n_cols + radius * 2);

	//fill center
	for (unsigned x = 0; x < src_image.n_cols; ++x)
		for (unsigned y = 0; y < src_image.n_rows; ++y)
			result(y + radius, x + radius) = src_image(y, x);
	
	//fill top and bottom
	for (unsigned y = radius; y < src_image.n_rows + radius; ++y)
		for (unsigned x = 0; x < radius; ++x)
		{
			result(y, x) = src_image(y - radius, radius - x);
			result(y, x + src_image.n_cols + radius) = src_image(y - radius, src_image.n_cols - x - 1);
		};

	//fill left and right side
	for (unsigned x = radius; x < src_image.n_cols + radius; ++x)
		for (unsigned y = 0; y < radius; ++y)
		{
			result(y, x) = src_image(radius - y, x - radius);
			result(y + src_image.n_rows + radius, x) = src_image(src_image.n_rows - y - 1, x - radius);
		};

	//fill corners
	int x1, x2, y1, y2;
	for (unsigned x = 0; x < radius; ++x)
		for (unsigned y = 0; y < radius; ++y)
		{
			x1 = radius - x - 1;
			y1 = radius - y - 1;
			x2 = src_image.n_cols - x - 1;
			y2 = src_image.n_rows - y - 1;
			result(y, x) = src_image(y1, x1); 				//left top
			result(y + src_image.n_rows + radius, x) = src_image(y2, x1);   //left bottom
			result(y, x + src_image.n_cols + radius) = src_image(y1,x2);    //right top
			result(y + src_image.n_rows + radius, x + src_image.n_cols + radius) = src_image(y2, x2); //right bottom
		}

	return result;
}

Image median_linear(Image src_image, int radius) {
	return src_image;
}

Image median_const(Image src_image, int radius) {
    return src_image;
}

void canny_pre(Image src_image, int threshold1, int threshold2, bool flag, int & X, int & Y);

Image canny(Image src_image, int threshold1, int threshold2)
{
	int x_l = 0, y_t = 0, x_r = 0, y_b = 0;
	canny_pre(src_image, threshold1, threshold2, 1, x_l, y_b);
	canny_pre(src_image, threshold1, threshold2, 0, x_r, y_t);
	return  src_image.submatrix(y_t, x_l, y_b - y_t, x_r - x_l);
}
	
void canny_pre(Image src_image, int threshold1, int threshold2, bool flag, int & X, int & Y) {
	unsigned row_n = src_image.n_rows; //image rows count
	unsigned col_n = src_image.n_cols; //image columns count
	Matrix <double> 	gradient(row_n, col_n); // gradient matrix
	Matrix <double> 	angle(row_n, col_n); //angles matrix
	Matrix <unsigned> 	tmp(row_n, col_n); //pre result matrix
	Matrix <bool> 		checked(row_n, col_n); //this matrix control single pass
	Image pre_res(row_n, col_n);

	//If flag = 1 we get left x and bottom Y cordinats
	//If flag = 0 we get right x and top Y cordinates
	//Inizialize arrays with zeroes
	for (unsigned x = 0; x < col_n; ++x)
		for (unsigned y = 0; y < row_n; ++y)
		{
			gradient(y, x) = 0;
			angle(y, x) = 0;
		};

	Image result = src_image.deep_copy();

	result = gaussian(result, 1.4, 2);

	Image Ix, Iy;
	
	if (flag)
	{
		Ix = sobel_x(result); // X derivative
		Iy = sobel_y(result); // Y derivative
	}
	else
	{
		Ix = sobel_x1(result); //I multiply soble kernel on -1
		Iy = sobel_y1(result); //Here too
	};


	//Calculate gradient and angle
	for (unsigned x = 0; x < col_n; ++x)
		for (unsigned y = 0; y < row_n; ++y)
		{
			gradient(y, x) = pow(pow(get<2>(Ix(y, x)), 2) + pow(get<2>(Iy(y, x)), 2), (0.5));
			angle(y, x) = atan2(get<2>(Iy(y, x)), get<2>(Ix(y, x)));
		}	

	//Not maximum suppression
	int x_off, y_off;

	for (unsigned x = 0; x < col_n; ++x)
		for (unsigned y = 0; y < row_n; ++y)
		{
			if ((x == 0 ) || (x == (col_n - 1)) || (y == 0) || (y == (row_n - 1)))
				continue;
			x_off = -1; y_off = 0;
			if ((angle(y, x) < 1 * (M_PI / 8)) 	&& (angle(y, x) >= 1 * (-M_PI / 8))) 	{ x_off = 1; y_off = 0; };
			if ((angle(y, x) < 3 * (M_PI / 8)) 	&& (angle(y, x) >= 1 * (M_PI / 8))) 	{ x_off = 1; y_off = 1; };
			if ((angle(y, x) < 5 * (M_PI / 8))	&& (angle(y, x) >= 3 * (M_PI / 8))) 	{ x_off = 0; y_off = 1; };
			if ((angle(y, x) < 7 * (M_PI / 8)) 	&& (angle(y, x) >= 5 * (M_PI / 8))) 	{ x_off = -1; y_off = 1; };

			if ((angle(y, x) < 5 * (- M_PI / 8)) 	&& (angle(y, x) >= 7 * (-M_PI / 8))) 	{ x_off = -1; y_off = -1; };
			if ((angle(y, x) < 3 * (- M_PI / 8))	&& (angle(y, x) >= 5 * (-M_PI / 8)))	{ x_off = 0; y_off = -1; };
			if ((angle(y, x) < 1 * (- M_PI / 8)) 	&& (angle(y, x) >= 3 * (-M_PI / 8))) 	{ x_off = 1; y_off = -1; };
		
			if ((abs(gradient(y, x)) <= abs(gradient(y + y_off, x + x_off))) ||		
			    (abs(gradient(y, x)) <= abs(gradient(y - y_off, x - x_off))))
				gradient(y, x) = 0;
		}
	
	//Clipping with threshold
	for (unsigned x = 0; x < col_n; ++x)
		for (unsigned y = 0; y < row_n; ++y)
		{
			tmp(y, x) = 1; // temporary value 0 - nothing; 1 - weak; 2 - strong 
			if (abs(gradient(y, x)) > threshold2) tmp(y, x) = 2;
			if (abs(gradient(y, x)) < threshold1) tmp(y, x) = 0;
			get<0>(pre_res(y, x)) = 125 * tmp(y, x);
			get<1>(pre_res(y, x)) = 125 * tmp(y, x);
			get<2>(pre_res(y, x)) = 125 * tmp(y, x);
		}

	//Tracking border
	
	//inzialize matrix
	for (unsigned x = 0; x < col_n; ++x)
		for (unsigned y = 0; y < row_n; ++y)
			checked(y, x) = false; // is item already be passed

	//Making border matrix
	{
		//inzialize local variable
		bool value = false; //is class strong
		std::vector <tuple<unsigned, unsigned>> add;
		
		for (unsigned x = 0; x < col_n; ++x)
			for (unsigned y = 0; y < row_n; ++y)
				if (!checked(y, x))
				{
					add.clear();
					checked(y, x) = true;
					value = examine_surround(y, x, checked, tmp, add);
					update(tmp, add, value);
				}						
	}

	//Calculate sum in each row and column
	int row_sum [row_n], col_sum[col_n];

	for (unsigned i = 0; i < row_n; ++i)
		row_sum[i] = 0;

	for (unsigned j = 0; j < col_n; ++j)
		col_sum[j] = 0;

	for (unsigned i = 0; i < row_n; ++i)
		for (unsigned j = 0; j < col_n; ++j)
		{
			row_sum[i] += tmp(i, j);
			col_sum[j] += tmp(i, j);
		}
	
	//////////Calculate 5 percent offset///////////

	int y_t, y_b, x_l, x_r; 
	
	x_l = border(col_sum, col_n, 0);
	x_r = border(col_sum, col_n, 1);
	y_t = border(row_sum, row_n, 0);
	y_b = border(row_sum, row_n, 1);

	if (flag)
	{
		X = x_l;
		Y = y_b;
	}
	else
	{
		X = x_r;
		Y = y_t;
	};
}

int max(int * array, int size, int l_bound, int r_bound)
{
	int max = array[l_bound], pos = l_bound;
    	for (int i = l_bound ; i < r_bound ; ++i)
	{
		if (array[i] > max)
		{
			pos = i;
			max = array[i];
		};
	}
	return pos;
}

void update(Matrix <unsigned> tmp, vector <tuple <unsigned, unsigned>> add, bool value)
{
	for (auto i = add.begin(); i != add.end(); ++i)
		tmp(get<0>(*i), get<1>(*i)) = int(value);
}

bool out_border(int a, int b, int N, int M)
{
	if ((a < 0) || (b < 0) || (a >= N) || b >= M) return true;
	return false;
}

bool examine_surround(int y, int x, Matrix <bool> check, Matrix <unsigned> tmp, vector <tuple <unsigned, unsigned>> add)
{
	bool flag = false;

	for (int i = -1; i < 2; ++i)
		for (int j = -1; j < 2; ++j)
		{	
			
			if (out_border(y + i, x + j, tmp.n_rows, tmp.n_cols)) continue; // out of border
			if (check(y + i,x + j)) continue; //already checked
			check(y + i, x + j) = true; //match that checked
			
			int val = tmp(y + i, x + j); 
			if (val != 0)
			{
				add.push_back(make_tuple(y + i, x + j));
				if (val == 2) flag = true;
				flag = flag || examine_surround(y + i, x+ j, check, tmp, add);
			}
		}
	return flag;
}

int min(int x, int y) 
{ 
	if (x > y) 
		return y;
	else 
		return x; 
}

int max(int x, int y)
{
	if (x < y)
		return y;
	else
		return x; 
}

int border(int * array, int size, bool flag)
{
	const float N = 0.10;
	int l_bound, r_bound;
	if (flag)
	{
		r_bound = size;
		l_bound = (1 - N) * size;
	}
	else
	{
		l_bound = 0;
		r_bound = N * size;
	};	

	int m1 = max(array, size, l_bound, r_bound);
	
	if (m1 != 0) array[m1 - 1] = 0;
	
	array[m1] = 0;

	if (m1 != size -1) array[m1 + 1] = 0;
	int m2 = max(array, size, l_bound, r_bound);

	if (flag)
		return min(m1, m2);
	else
		return max(m1, m2);
}
