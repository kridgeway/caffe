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
  ssim.Reshape(shape);
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
  // diff_ = a - b
  caffe_sub(
    count,
    bottom[0]->cpu_data(),
    bottom[1]->cpu_data(),
    diff_.mutable_cpu_data());

  size_t dim = (size_t)(bottom[0]->count() / bottom[0]->num());
  int width = (int)( sqrt(dim/3) );
  int height = width;
  int nChan=3, d=IPL_DEPTH_32F;
  int imageSize = width*height*nChan;
  const Dtype* bottom0data = bottom[0]->cpu_data();
  const Dtype* bottom1data = bottom[1]->cpu_data();

  for( size_t image_idx = 0; image_idx < bottom[0]->num(); image_idx++ ) {
    const Dtype* img1_data = bottom0data + image_idx*imageSize;
    const Dtype* img2_data = bottom1data + image_idx*imageSize;
    Dtype* target = ssim_data_.mutable_cpu_data() + image_idx*imageSize;
    //ssim.debug = image_idx == 0;
    ssim.CalculateSSIM(img1_data, img2_data, target);
    // Rescale to [0,1], find 1 - ssim
    caffe_scal(
      imageSize,
      Dtype(-0.5),
      target);
    caffe_add_scalar(
      imageSize,
      Dtype(0.5),
      target);
    //if( image_idx == 0 )
    //  printf("%f\n", target[0]);
    // Set the sign equal to the sign of the L1 diff
    Dtype* diffData = diff_.mutable_cpu_data() + image_idx * imageSize;
    for( int didx=0; didx < dim; didx++ ) {
      target[didx] = target[didx] * (Dtype)caffe_sign( diffData[didx] );
    }
  }
  Dtype dot = caffe_cpu_dot(count,
    ssim_data_.cpu_data(),
    ssim_data_.cpu_data()
  );
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
        bottom[i]->count(),              // count
        alpha,                           // alpha
        ssim_data_.cpu_data(),           // a
        Dtype(0),                        // beta
        bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(SSIMLossLayer);
REGISTER_LAYER_CLASS(SSIMLoss);

}
