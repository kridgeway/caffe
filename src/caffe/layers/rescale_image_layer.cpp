#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
RescaleImageLayer<Dtype>::RescaleImageLayer(const LayerParameter& param) : 
  Layer<Dtype>(param) {
  has_data_mean_ = false;
  CHECK(param.has_rescale_image_param());
  scale_ = param.rescale_image_param().scale();
  if (param.rescale_image_param().has_mean_file() ) {
    const std::string mean_file = param.rescale_image_param().mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    LOG(INFO) << "Scaling by " << scale_;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    caffe_scal( data_mean_.count(),
        scale_,
        data_mean_.mutable_cpu_data() );
    vector<int> flatShape(2);
    flatShape[0] = 1;
    flatShape[1] = data_mean_.count();
    data_mean_flat_.Reshape( flatShape );
    data_mean_flat_.ShareData( data_mean_ );
    has_data_mean_ = true;
  } else if( param.rescale_image_param().has_mean_value() ) {
    mean_value_ = param.rescale_image_param().mean_value();
    mean_value_ = mean_value_ * scale_;
  } else {
    mean_value_ = 0.0f;
  }
}

template <typename Dtype>
void RescaleImageLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size());
  for( int idx = 0; idx < top.size(); idx++ ) {
    top[idx]->ReshapeLike( *bottom[idx] );
  }
}


template <typename Dtype>
void RescaleImageLayer<Dtype>::Rescale( Blob<Dtype>* source, Blob<Dtype>* target ) {
  int d = source->count() / source->num();
  caffe_copy(source->count(), source->cpu_data(), target->mutable_cpu_data());
  for( int idx =0; idx < source->num(); idx++ ) {
    // add the mean back in
    if( has_data_mean_ ) {
      printf("add data mean\n");
      caffe_add(
          d,
          source->cpu_data()+idx*d,
          data_mean_flat_.cpu_data(),
          target->mutable_cpu_data()+idx*d
      );
    } else {
       caffe_add_scalar(
           d,
           mean_value_,
           target->mutable_cpu_data()+idx*d);
    }
    caffe_scal(
        d,
        Dtype(1.0) / scale_,
        target->mutable_cpu_data()+idx*d
    );
  }
}

template <typename Dtype>
void RescaleImageLayer<Dtype>::UnRescale( Blob<Dtype>* source, Blob<Dtype>* target ) {
  int d = source->count() / source->num();
  caffe_copy(source->count(), source->cpu_diff(), target->mutable_cpu_diff());
  for( int idx =0; idx < source->num(); idx++ ) {
    //unscale
    caffe_scal(
        d,
        scale_,
        target->mutable_cpu_diff()+idx*d
    );
  }
}

template <typename Dtype>
void RescaleImageLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for( int idx=0; idx < bottom.size(); idx++ ) {
    Rescale( bottom[idx], top[idx] );
  }
}

template <typename Dtype>
void RescaleImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      UnRescale( top[i], bottom[i] );
    }
  }
}

INSTANTIATE_CLASS(RescaleImageLayer);
REGISTER_LAYER_CLASS(RescaleImage);

}
