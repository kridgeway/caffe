#include <cv.h>
#include "caffe/loss_layers.hpp"

#include <highgui.h>
#include <vector>

namespace caffe {

template<typename Dtype>
SSIM<Dtype>::SSIM() { }

template<typename Dtype>
SSIM<Dtype>::~SSIM() {
  cvReleaseImageHeader(&img1);
  cvReleaseImageHeader(&img2);
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
}

template<typename Dtype>
void SSIM<Dtype>::LayerSetUp(size_t width, size_t height, size_t nChan) {
  height_ = height;
  width_ = width;
  nChan_ = nChan;
  int d = IPL_DEPTH_32F;
  CvSize size = cvSize(width_, height_);

  //img1 = cvCreateImage( size, d, nChan);
  //img2 = cvCreateImage( size, d, nChan);
  img1 = cvCreateImageHeader(size, d, nChan);
  img2 = cvCreateImageHeader(size, d, nChan);
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
}

template<typename Dtype>
void SSIM<Dtype>::Reshape(std::vector<int> shape) {
  img1_reformatted_.Reshape(shape);
  img2_reformatted_.Reshape(shape);
}

template<typename Dtype>
void SSIM<Dtype>::CalculateSSIM(const Dtype *img1_data,
                                const Dtype *img2_data,
                                Dtype *target) {
  double C1 = 6.5025, C2 = 58.5225;
  Dtype *img1_data_rfmt = img1_reformatted_.mutable_cpu_data();
  Dtype *img2_data_rfmt = img2_reformatted_.mutable_cpu_data();

  // step 1, get image data into img1 and img2
  //TODO this is the worst
  for (int chan_idx = 0; chan_idx < nChan_; chan_idx++) {
    for (int row_idx = 0; row_idx < height_; row_idx++) {
      for (int col_idx = 0; col_idx < width_; col_idx++) {
        int source = chan_idx * width_ * height_ + row_idx * width_ + col_idx;
        int target = row_idx * width_ * nChan_ + col_idx * nChan_ + chan_idx;
        img1_data_rfmt[target] = img1_data[source];
        img2_data_rfmt[target] = img2_data[source];
      }
    }
  }
  cvSetData(img1, (void *) img1_data_rfmt, img1->widthStep);
  cvSetData(img2, (void *) img2_data_rfmt, img1->widthStep);

  /**
  IplImage  *out  = cvCreateImage( cvSize(width,height), IPL_DEPTH_8U, 3);
  cvCvtScale(img1,out,1,0);
  cvSaveImage("img1.png",out);
  cvCvtScale(img2,out,1,0);
  cvSaveImage("img2.png",out);
  cvReleaseImage(&out);
  */

  // Step 2, calculate ssim_map
  cvPow(img1, img1_sq, 2);
  cvPow(img2, img2_sq, 2);
  cvMul(img1, img2, img1_img2, 1);

  cvSmooth(img1, mu1, CV_GAUSSIAN, 11, 11, 1.5);
  cvSmooth(img2, mu2, CV_GAUSSIAN, 11, 11, 1.5);

  cvPow(mu1, mu1_sq, 2);
  cvPow(mu2, mu2_sq, 2);
  cvMul(mu1, mu2, mu1_mu2, 1);

  cvSmooth(img1_sq, sigma1_sq, CV_GAUSSIAN, 11, 11, 1.5);
  cvAddWeighted(sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq);

  cvSmooth(img2_sq, sigma2_sq, CV_GAUSSIAN, 11, 11, 1.5);
  cvAddWeighted(sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq);

  cvSmooth(img1_img2, sigma12, CV_GAUSSIAN, 11, 11, 1.5);
  cvAddWeighted(sigma12, 1, mu1_mu2, -1, 0, sigma12);

  cvScale(mu1_mu2, temp1, 2);
  cvAddS(temp1, cvScalarAll(C1), temp1);

  // (2*sigma12 + C2)
  cvScale(sigma12, temp2, 2);
  cvAddS(temp2, cvScalarAll(C2), temp2);

  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
  cvMul(temp1, temp2, temp3, 1);

  // (mu1_sq + mu2_sq + C1)
  cvAdd(mu1_sq, mu2_sq, temp1);
  cvAddS(temp1, cvScalarAll(C1), temp1);

  // (sigma1_sq + sigma2_sq + C2)
  cvAdd(sigma1_sq, sigma2_sq, temp2);
  cvAddS(temp2, cvScalarAll(C2), temp2);

  // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvMul(temp1, temp2, temp1, 1);

  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvDiv(temp3, temp1, ssim_map, 1);

  //CvScalar index_scalar = cvAvg( ssim_map );
  //printf("SSIM r=%.2f g=%.2f b=%.2f\n", index_scalar.val[2] * 100, index_scalar.val[1] * 100, index_scalar.val[0] * 100 );

  // Step 3 copy ssim_map to top
  Dtype *data = (Dtype *) ssim_map->imageData;

  size_t imageSize = (size_t) width_ * height_ * nChan_;
  memcpy(target, data, imageSize * sizeof(Dtype));
}

INSTANTIATE_CLASS(SSIM);

}
