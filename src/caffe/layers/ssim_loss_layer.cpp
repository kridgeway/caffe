#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void SSIMLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1)) << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1)) << "Weight data must have same dimension";
  diff_.ReshapeLike(*bottom[0]);
  ssim_data_.ReshapeLike(*bottom[0]);
  vector<int> shape = bottom[0]->shape();
  shape[0] = 1;
  if( sizeof(Dtype) != sizeof(float) ) {
    throw std::runtime_error("SSIM layer only supports float");
  }
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  size_t dim = (size_t)(bottom[0]->count() / bottom[0]->num());
  //TODO this assumes square images with 3 channels, which might not be the case
  int x = (int)floor( sqrt(dim/3) );
  int y = x;
  int nChan=3;
  ssim.LayerSetUp(x,y, nChan);
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  size_t dim = (size_t)(bottom[0]->count() / bottom[0]->num());
  int width = (int)( sqrt(dim/3) );
  int height = width;
  int nChan=3;
  int imageSize = width*height*nChan;
  const Dtype* bottom0data = bottom[0]->cpu_data();
  const Dtype* bottom1data = bottom[1]->cpu_data();
  //Dtype* topData = top[0]->mutable_cpu_data();
  Dtype* topData = ssim_data_.mutable_cpu_data();
  Dtype* gradientData = diff_.mutable_cpu_data();

  for( size_t image_idx = 0; image_idx < bottom[0]->num(); image_idx++ ) {
    const Dtype* img1_data = bottom0data + image_idx*imageSize;
    const Dtype* img2_data = bottom1data + image_idx*imageSize;
    Dtype* target = topData + image_idx*imageSize;
    Dtype* target_gradient = gradientData + image_idx*imageSize;
    ssim.debug = image_idx == 0;
    ssim.CalculateSSIM((float*)img1_data, (float*)img2_data, (float*)target, (float*)target_gradient);
  }
  Dtype sumSSIM = caffe_cpu_asum(count, ssim_data_.cpu_data() );
  Dtype loss =  sumSSIM / bottom[0]->count();
  top[0]->mutable_cpu_data()[0] = loss;
  //printf("img1[0] = %f img2[0] = %f\n", bottom0data[0], bottom1data[0]);
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      //const Dtype sign = (i == 0) ? -1 : 1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      //if( i == 1 )
      //  printf("Delta %f alpha = %f\n", bottom[i]->mutable_cpu_diff()[0], alpha );
      caffe_cpu_axpby(
        bottom[i]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),
        bottom[i]->mutable_cpu_diff());
      //if( i == 1 )
      //  printf("Delta after %f\n",  bottom[i]->mutable_cpu_diff()[0] );
    }
  }
}

INSTANTIATE_CLASS(SSIMLossLayer);
REGISTER_LAYER_CLASS(SSIMLoss);

}
