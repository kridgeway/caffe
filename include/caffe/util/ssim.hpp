#pragma once
#include <cv.h>
#include <highgui.h>
#include <vector>

typedef struct _IplImage IplImage;

namespace caffe {
class SSIM {
public:
  bool debug;
  SSIM();

  virtual ~SSIM();

  virtual void LayerSetUp(int x, int y, int nChan);
  virtual void caffeToCV(std::vector<const float*>& caffeData,
                         std::vector<float*>& cvData, bool convert );
  virtual void cvToCaffe(std::vector<const float*>& cvData,
                         std::vector<float*>& caffeData, bool convert);
  virtual void CalculateSSIM(const float *img1_data,
                             const float *img2_data,
                             float *target, float* target_gradient=NULL,
                             bool convertCVCaffe=true);

protected:
  IplImage
    *img1, *img2, *img1_img2,
    *img1_sq, *img2_sq,
    *mu1, *mu2,
    *mu1_sq, *mu2_sq, *mu1_mu2,
    *sigma1_sq, *sigma2_sq, *sigma12,
    *a1_a2,*b1_b2, *a1, *a2, *b1, *b2,
    *ssim_map, *temp1, *temp2, *temp3, *gradient;
  int nChan_;
  int height_;
  int width_;

  void ssimGradient();
};
}
