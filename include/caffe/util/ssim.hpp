#pragma once
#include <cv.h>
#include <highgui.h>
#include <vector>

typedef struct _IplImage IplImage;

namespace caffe {

template<typename Dtype>
class SSIM {
public:
  SSIM();

  virtual ~SSIM();

  virtual void LayerSetUp(size_t x, size_t y, size_t nChan);

  virtual void Reshape(std::vector<int> shape);

  virtual void CalculateSSIM(const Dtype *img1_data,
                             const Dtype *img2_data, Dtype *target);

protected:
  IplImage
    *img1, *img2, *img1_img2,
    *img1_sq, *img2_sq,
    *mu1, *mu2,
    *mu1_sq, *mu2_sq, *mu1_mu2,
    *sigma1_sq, *sigma2_sq, *sigma12,
    *ssim_map, *temp1, *temp2, *temp3;
  caffe::Blob<Dtype> img1_reformatted_;
  caffe::Blob<Dtype> img2_reformatted_;
  size_t nChan_;
  size_t height_;
  size_t width_;
};
}
