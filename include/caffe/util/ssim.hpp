#pragma once
#include <cv.h>
#include <highgui.h>
#include <vector>

typedef struct _IplImage IplImage;

namespace caffe {
template<typename Dtype>
class SSIM {
public:
  bool debug;
  SSIM();

  virtual ~SSIM();

  virtual void LayerSetUp(int x, int y, int nChan);

  virtual void Reshape(std::vector<int> shape);

  virtual void CalculateSSIM(const Dtype *img1_data,
                             const Dtype *img2_data,
                             Dtype *target, Dtype* target_gradient=NULL);

protected:
  IplImage
    *img1, *img2, *img1_img2,
    *img1_sq, *img2_sq,
    *mu1, *mu2,
    *mu1_sq, *mu2_sq, *mu1_mu2,
    *sigma1_sq, *sigma2_sq, *sigma12,
    *a1_a2,*b1_b2, *a1, *a2, *b1, *b2,
    *ssim_map, *temp1, *temp2, *temp3, *gradient;
  caffe::Blob<Dtype> img1_reformatted_;
  caffe::Blob<Dtype> img2_reformatted_;
  int nChan_;
  int height_;
  int width_;

  void ssimGradient();
};
}
