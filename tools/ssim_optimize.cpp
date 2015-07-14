#include <cv.h>
#include <highgui.h>
#include "caffe/util/ssim.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;
using namespace caffe;


void printImg( IplImage* img, int channel ) {
  float* data = (float*)img->imageData;
  for( int col = 0; col < img->width; col++ ) {
    for( int row = 0; row < img->height; row++ ) {
      int idx = row * img->height * 3 + col * 3 + channel;
      printf("%.15f ", data[idx] );
    }
  }
  printf("\n");
}

int main( int argc, char** argv) {
  IplImage* img1_temp = cvLoadImage(argv[1]);
  IplImage* img2_temp = cvLoadImage(argv[2]);

  int nChan=3, d=IPL_DEPTH_32F;
  CvSize size = cvSize(32, 32);

  IplImage* img1 = cvCreateImage( size, d, nChan);
  IplImage* img2 = cvCreateImage( size, d, nChan);

  IplImage* gradient = cvCreateImage(size,d,nChan);
  IplImage* ssim_map = cvCreateImage(size,d,nChan);

  cvConvert(img1_temp, img1);
  cvConvert(img2_temp, img2);

  SSIM ssim;
  ssim.LayerSetUp(32,32,3);

  for( int i = 0; i < 5000; i++ ) {
    if( i == 4999 )
      ssim.debug=1;
    else
      ssim.debug=0;
    ssim.CalculateSSIM((float *) img1->imageData,
                       (float *) img2->imageData,
                       (float*) ssim_map->imageData,
                       (float*) gradient->imageData, false);
    float gSum = caffe_cpu_asum(32*32*3, (float*)gradient->imageData );
//    cvScale(gradient, gradient, 50);
    cvAdd(gradient, img2, img2);
    if (i % 100 == 0) {
      //printImg(img2, 1);
      CvScalar index_scalar = cvAvg(ssim_map);
      cout << "(R, G & B SSIM index)" << endl;
      cout << index_scalar.val[2] * 100 << "%" << endl;
      cout << index_scalar.val[1] * 100 << "%" << endl;
      cout << index_scalar.val[0] * 100 << "%" << endl;

      cout << ((float*)img1->imageData)[0]<< " " << ((float*)img2->imageData)[0]<<endl;

      cout << caffe_cpu_asum(32*32*3, (float*)img1->imageData) << " "<< caffe_cpu_asum(32*32*3,(float*)img2->imageData) << " "<<
        gSum << endl;
    }
  }
  return 0;
}