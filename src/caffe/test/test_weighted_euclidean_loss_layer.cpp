#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "../../../include/caffe/filler.hpp"

namespace caffe {
template <typename TypeParam>
class WeightedEuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    protected:
    WeightedEuclideanLossLayerTest()
        : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
          blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
          blob_bottom_weights_(new Blob<Dtype>(1, 5,1,1)),
          blob_top_loss_(new Blob<Dtype>()) {
        // fill the values
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_data_);

        filler.Fill(this->blob_bottom_label_);
        blob_bottom_vec_.push_back(blob_bottom_label_);

        FillerParameter weightFillerParam;
        weightFillerParam.set_min(0.0f);
        weightFillerParam.set_max(1.0f);
        UniformFiller<Dtype> weightFiller(weightFillerParam);
        weightFiller.Fill(this->blob_bottom_weights_);

        printf("Num %d count %d\n", blob_bottom_weights_->num(),
               blob_bottom_weights_->count());

        blob_bottom_vec_.push_back(blob_bottom_weights_);

        blob_top_vec_.push_back(blob_top_loss_);
        printf("Blob 0\n");
        printBlob( blob_bottom_data_ );
        printf("Blob 1\n");
        printBlob( blob_bottom_label_ );
        printf("Weights\n");
        printBlob( blob_bottom_weights_ );
    }
    virtual ~WeightedEuclideanLossLayerTest() {
        delete blob_bottom_data_;
        delete blob_bottom_label_;
        delete blob_bottom_weights_;
        delete blob_top_loss_;
    }
    void TestForward() {
        // Get the loss without a specified objective weight -- should be
        // equivalent to explicitly specifiying a weight of 1.
        LayerParameter layer_param;
        WeightedEuclideanLossLayer<Dtype> layer_weight_1(layer_param);
        layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype loss_weight_1 =
            layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        printf("Blob 0\n");
        printBlob( blob_bottom_data_ );
        printf("Blob 1\n");
        printBlob( blob_bottom_label_ );
        printf("Weights\n");
        printBlob( blob_bottom_weights_ );
        printf("Loss\n");
        printBlob( blob_top_loss_ );
    }

    void printBlob( Blob<Dtype>* blob ) {
        for( size_t idx = 0; idx < blob->num(); idx++ ) {
            size_t d = blob->count() / blob->num();
            const Dtype* data = blob->cpu_data() + idx * d;
            for( size_t jj = 0; jj < d; jj++ ) {
                printf("%.4f ",  data[jj] );
            }
            printf("\n");
        }
    }
    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_label_;
    Blob<Dtype>* const blob_bottom_weights_;
    Blob<Dtype>* const blob_top_loss_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};
TYPED_TEST_CASE(WeightedEuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(WeightedEuclideanLossLayerTest, TestForward) {
  this->TestForward();
}
}
