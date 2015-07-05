#include <cv.h>
#include "caffe/loss_layers.hpp"

#include <highgui.h>
#include <vector>

#define WINDOW_SIZE 11

namespace caffe {

SSIM::SSIM() { debug = false; }

SSIM::~SSIM() {
  cvReleaseImage(&img1);
  cvReleaseImage(&img2);
  cvReleaseImage(&img1_sq);
  cvReleaseImage(&img2_sq);
  cvReleaseImage(&img1_img2);
  cvReleaseImage(&mu1);
  cvReleaseImage(&mu2);
  cvReleaseImage(&mu1_sq);
  cvReleaseImage(&mu2_sq);
  cvReleaseImage(&mu1_mu2);
  cvReleaseImage(&sigma1_sq);
  cvReleaseImage(&sigma2_sq);
  cvReleaseImage(&sigma12);
  cvReleaseImage(&temp1);
  cvReleaseImage(&temp2);
  cvReleaseImage(&temp3);
  cvReleaseImage(&ssim_map);
  cvReleaseImage(&a1_a2);
  cvReleaseImage(&b1_b2);
  cvReleaseImage(&a1);
  cvReleaseImage(&a2);
  cvReleaseImage(&b1);
  cvReleaseImage(&b2);
  cvReleaseImage(&gradient);
}

void SSIM::LayerSetUp(int width, int height, int nChan) {
  height_ = height;
  width_ = width;
  nChan_ = nChan;
  int d = IPL_DEPTH_32F;
  CvSize size = cvSize(width_, height_);

  img1 = cvCreateImage( size, d, nChan);
  img2 = cvCreateImage( size, d, nChan);
  img1_sq = cvCreateImage(size, d, nChan);
  img2_sq = cvCreateImage(size, d, nChan);
  img1_img2 = cvCreateImage(size, d, nChan);
  mu1 = cvCreateImage(size, d, nChan);
  mu2 = cvCreateImage(size, d, nChan);
  mu1_sq = cvCreateImage(size, d, nChan);
  mu2_sq = cvCreateImage(size, d, nChan);
  mu1_mu2 = cvCreateImage(size, d, nChan);
  sigma1_sq = cvCreateImage(size, d, nChan);
  sigma2_sq = cvCreateImage(size, d, nChan);
  sigma12 = cvCreateImage(size, d, nChan);
  temp1 = cvCreateImage(size, d, nChan);
  temp2 = cvCreateImage(size, d, nChan);
  temp3 = cvCreateImage(size, d, nChan);
  ssim_map = cvCreateImage(size, d, nChan);
  a1_a2= cvCreateImage(size, d, nChan);
  b1_b2 = cvCreateImage(size, d, nChan);
  a1 = cvCreateImage(size, d, nChan);
  a2 = cvCreateImage(size, d, nChan);
  b1 = cvCreateImage(size, d, nChan);
  b2 = cvCreateImage(size, d, nChan);
  gradient = cvCreateImage(size, d, nChan);
}

void saveValues( IplImage* img, std::string filename ) {
  FILE* f = fopen(filename.c_str(),"w");
  int channel = 1;
  float* data = (float*)img->imageData;
  for( int col = 0; col < img->width; col++ ) {
    for( int row = 0; row < img->height; row++ ) {
      int idx = row * img->height * 3 + col * 3 + channel;
      fprintf(f, "%.15f ", data[idx] );
    }
  }
  fclose(f);
}

void saveImage( IplImage* img, std::string filename) {
  IplImage  *out  = cvCreateImage( cvSize(img->width,img->height), IPL_DEPTH_8U, 3);
  cvCvtScale(img,out,1,0);
  cvSaveImage(filename.c_str(),out);
  cvReleaseImage(&out);
}

void SSIM::caffeToCV(std::vector<const float*>& caffeData, std::vector<float*>& cvData, bool convert ) {
  for (int chan_idx = 0; chan_idx < nChan_; chan_idx++) {
    for (int row_idx = 0; row_idx < height_; row_idx++) {
      for (int col_idx = 0; col_idx < width_; col_idx++) {
        int source;
        int target = row_idx * width_ * nChan_ + col_idx * nChan_ + chan_idx;
        if (convert) {
          source = chan_idx * width_ * height_ + row_idx * width_ + col_idx;
        } else {
          source = target;
        }
        for( size_t idx = 0; idx < caffeData.size(); idx++ ) {
          cvData[idx][target] = caffeData[idx][source];
        }
      }
    }
  }
}
void SSIM::cvToCaffe(std::vector<const float*>& cvData, std::vector<float*>& caffeData, bool convert ) {
  for( int chan_idx = 0; chan_idx < nChan_; chan_idx++ ) {
    for( int row_idx = 0; row_idx < height_; row_idx++) {
      for(int col_idx = 0; col_idx < width_; col_idx++ ) {
        int target_idx;
        int source_idx = row_idx * width_ * nChan_ + col_idx * nChan_ + chan_idx;
        if( convert ) {
          target_idx = chan_idx * width_ * height_ + row_idx * width_ + col_idx;
        } else {
          target_idx = source_idx;
        }
        for( size_t idx = 0; idx < caffeData.size(); idx++ ) {
          caffeData[idx][target_idx] = cvData[idx][source_idx];
        }
      }
    }
  }
}

void SSIM::CalculateSSIM(const float *img1_data,
                                const float *img2_data,
                                float *target,
                                float* target_gradient,
                                bool convertCVCaffe) {
  double C1 = 6.5025, C2 = 58.5225;
  vector<const float *> from;
  vector<float *> to;
  from.push_back(img1_data);
  from.push_back(img2_data);
  to.push_back((float *) img1->imageData);
  to.push_back((float *) img2->imageData);
  caffeToCV(from,to, convertCVCaffe);

  // Step 2, calculate ssim_map
  cvPow(img1, img1_sq, 2);
  cvPow(img2, img2_sq, 2);
  cvMul(img1, img2, img1_img2, 1);

  cvSmooth(img1, mu1, CV_GAUSSIAN, WINDOW_SIZE, WINDOW_SIZE, 1.5);
  cvSmooth(img2, mu2, CV_GAUSSIAN, WINDOW_SIZE, WINDOW_SIZE, 1.5);

  cvPow(mu1, mu1_sq, 2);
  cvPow(mu2, mu2_sq, 2);
  cvMul(mu1, mu2, mu1_mu2, 1);

  cvSmooth(img1_sq, sigma1_sq, CV_GAUSSIAN, WINDOW_SIZE, WINDOW_SIZE, 1.5);
  cvAddWeighted(sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq);

  cvSmooth(img2_sq, sigma2_sq, CV_GAUSSIAN, WINDOW_SIZE, WINDOW_SIZE, 1.5);
  cvAddWeighted(sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq);

  cvSmooth(img1_img2, sigma12, CV_GAUSSIAN, WINDOW_SIZE, WINDOW_SIZE, 1.5);
  cvAddWeighted(sigma12, 1, mu1_mu2, -1, 0, sigma12);

  cvScale( mu1_mu2, a1, 2 );
  cvAddS( a1, cvScalarAll(C1), a1 );

  // (2*sigma12 + C2)
  cvScale( sigma12, a2, 2 );
  cvAddS( a2, cvScalarAll(C2), a2 );

  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
  cvMul( a1, a2, a1_a2, 1 );

  // (mu1_sq + mu2_sq + C1)
  cvAdd( mu1_sq, mu2_sq, b1 );
  cvAddS( b1, cvScalarAll(C1), b1 );

  // (sigma1_sq + sigma2_sq + C2)
  cvAdd( sigma1_sq, sigma2_sq, b2 );
  cvAddS( b2, cvScalarAll(C2), b2 );

  // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvMul( b1, b2, b1_b2, 1 );

  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvDiv( a1_a2, b1_b2, ssim_map, 1 );

  bool compute_gradient = target_gradient != NULL;
  if( compute_gradient )
    ssimGradient();

  if( debug ) {
    saveImage( img1, "caffessim1.png");
    saveImage( img2, "caffessim2.png");
    saveValues( ssim_map, "caffessim_map.vec");
    saveValues( gradient, "caffessim_gradient.vec");
  }

  // Step 3 copy ssim_map to top
  float *data = (float *) ssim_map->imageData;
  float *gdata = (float *) gradient->imageData;

  from.clear();
  to.clear();
  from.push_back(data);
  to.push_back(target);
  if( target_gradient != NULL ) {
    from.push_back( gdata );
    to.push_back(target_gradient);
  }
  cvToCaffe(from,to, convertCVCaffe);
}

void SSIM::ssimGradient() {
  cvMul(b2,img1,temp1);      // temp1 = b_2 * img1
  cvMul(a2,img2,temp2);      // temp2 = a_2 * img2
  cvSub(temp1, temp2, temp1);  // temp1 = b_2*img1 - a_2*img2
  cvMul(a1, b1, temp2);      // temp2 = a_1 * b_1
  cvMul(temp2, temp1, temp1 );// temp1 = a_1 * b_1( b_2 * x - a_2 * y )

  //b1.*b2.*(a2-a1).*mu1 + ...
  cvSub(a2, a1, temp2);     // temp2 = a_2 - a_1
  cvMul(mu1, temp2, temp2);  // temp2 = mu_1 * (a_2 - a_1)
  cvMul(b1_b2, temp2, temp2); // temp2 = b1.*b2 .* mu_1 .* (a_2 - a_1)
  cvAdd(temp1, temp2, temp1); // temp1 = a_1 * b_1( b_2 * x - a_2 * y ) + b1.*b2 .* mu_1 .* (a_2 - a_1)

  //a1.*a2.*(b1-b2).*mu2 ...
  cvSub(b1, b2, temp2);     // temp2 = b_1 - b_2
  cvMul(mu2, temp2, temp2);  // temp2 = (b_1-b_2).*mu_2
  cvMul(a1_a2, temp2, temp2); // temp2 = a_1.*a_2*(b_1-b_2).*mu_2
  cvAdd(temp1,temp2,temp1);   // temp1 = a_1 * b_1( b_2 * x - a_2 * y ) + b1.*b2 .* mu_1 .* (a_2 - a_1) + a_1.*a_2*(b_1-b_2).*mu_2

  //p1 = (2 ./ ((b1.^2) .* (b2.^2) .* window_size));
  cvPow(b1, temp2, 2.0f);                  // temp2 = b_1^2
  cvPow(b2, temp3, 2.0f);                  // temp3 = b_2^2
  cvMul(temp2, temp3, temp2);               // temp2 = b_1^2 * b_2^2
  cvScale(temp2, temp2, float(WINDOW_SIZE));// temp2 = WINDOW_SIZE * (b_1^2 * b_2^2)
  cvPow(temp2, temp2, -1.0f);               // temp2 = 1 / WINDOW_SIZE * (b_1^2 * b_2^2);
  cvScale(temp2,temp2, 2.0f );              // temp2 = 2 / (WINDOW_SIZE * (b_1^2 * b_2^2));
  cvMul(temp2, temp1, gradient);
}

}
