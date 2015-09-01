#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"
#include <cv.h>	
#include <highgui.h>

namespace caffe {

template <typename Dtype>
SSIMLayer<Dtype>::SSIMLayer(const LayerParameter& param) : Layer<Dtype>(param) { }

template <typename Dtype>
void SSIMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //int x=img1_temp->width, y=img1_temp->height;
  size_t dim = (size_t)(bottom[0]->count() / bottom[0]->num());
  //TODO this assumes square images with 3 channels, which might not be the case
  std::vector<int> shape = bottom[0]->shape();
  int x= shape[2];
  int y= shape[3];
  int nChan= shape[1];
  int d=IPL_DEPTH_32F;
  ssim.LayerSetUp(x,y, nChan);
  if( sizeof(Dtype) != sizeof(float) ) {
    throw std::runtime_error("SSIM layer only supports float");
  }
}

template <typename Dtype>
SSIMLayer<Dtype>::~SSIMLayer() { }

template <typename Dtype>
void SSIMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> shape = bottom[0]->shape();
  shape[0] = 1;
}


template <typename Dtype>
void SSIMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  size_t dim = (size_t)(bottom[0]->count() / bottom[0]->num());
  std::vector<int> shape = bottom[0]->shape();
  int height= shape[2];
  int width= shape[3];
  int nChan= shape[1];
  int d=IPL_DEPTH_32F;
  size_t imageSize = (size_t) width*height*nChan;
  // default settings

  const Dtype* bottom0data = bottom[0]->cpu_data();
  const Dtype* bottom1data = bottom[1]->cpu_data();
  Dtype* topData = top[0]->mutable_cpu_data();

  for( size_t image_idx = 0; image_idx < bottom[0]->num(); image_idx++ ) {
    const Dtype* img1_data = bottom0data + image_idx*imageSize;
    const Dtype* img2_data = bottom1data + image_idx*imageSize;

    Dtype* target = topData + image_idx*imageSize;
    ssim.CalculateSSIM((float*)img1_data, (float*)img2_data, (float*)target);

    // Rescale to [0,1]
    caffe_scal(
        imageSize,
        Dtype(0.5),
        target);
    caffe_add_scalar(
        imageSize,
        Dtype(0.5),
        target);
  }
}

INSTANTIATE_CLASS(SSIMLayer);
REGISTER_LAYER_CLASS(SSIM);

}
