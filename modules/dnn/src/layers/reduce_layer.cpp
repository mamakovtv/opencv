// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/core/mat.hpp>

namespace cv { namespace dnn {

class ReduceLayerImpl CV_FINAL : public ReduceLayer
{
public:
    ReduceLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        keepDims = (params.get<int>("keepdims", 1) == 1);

        axes = params.get("axes");

        const std::string& typeStr = params.get<std::string>("type");

        if (typeStr == "L1")
        {
            type = ReduceType::L1;
        }
        else if (typeStr == "L2")
        {
            type = ReduceType::L2;
        }
        else
        {
            CV_Error(Error::StsBadArg, "Unsupported reduce type");
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV && preferableTarget == DNN_TARGET_CPU;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int /*requiredOutputs*/,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &/*internals*/) const CV_OVERRIDE
    {
        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            const auto& input = inputs[i];
            auto& output = outputs[i];

            auto nInpDims = input.size();
            std::vector<bool> dimsToKeep(nInpDims, true);
            std::vector<bool> dimsToReduce(nInpDims, false);
            for (int i = 0; i < axes.size(); i++)
            {
                int axis = normalize_axis(axes.get<int>(i), nInpDims);
                dimsToKeep[axis] = keepDims;
                dimsToReduce[axis] = true;
            }

            output.clear();
            for (auto i = 0; i < nInpDims; i++)
            {
                if (dimsToKeep[i])
                {
                    if (dimsToReduce[i])
                    {
                        output.push_back(1);
                    }
                    else
                    {
                        output.push_back(input[i]);
                    }
                }
                else
                {
                    CV_Assert(dimsToReduce[i]);
                }
            }
        }

        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_Assert_N(inputs.size() == 1, outputs.size() == 1);

        auto inpShape = shape(inputs[0]);
        int nInpDims = inpShape.size();
        std::vector<bool> dimsToKeep(nInpDims, true);
        std::vector<bool> dimsToReduce(nInpDims, false);
        for (int i = 0; i < axes.size(); i++)
        {
            int axis = normalize_axis(axes.get<int>(i), nInpDims);
            dimsToKeep[axis] = keepDims;
            dimsToReduce[axis] = true;
        }

        std::vector<MatShape> outShape;
        getMemoryShapes({shape(inputs[0])}, 0, outShape, outShape );
        CV_Assert(!outShape.empty());
        Mat mOut(outShape[0], CV_32SC1);

        std::vector<int> nDimInpId(inpShape.size(), 0);
        auto increaseNdimId = [&nDimInpId, &inpShape]()
        {
            nDimInpId.back()++;
            for (int i = nDimInpId.size() - 1; i >= 0 ; i--)
            {
                if (nDimInpId[i] >= inpShape[i])
                {
                    if (i==0)
                    {
                        return false;
                    }
                    nDimInpId[i] = 0;
                    nDimInpId[i-1]++;
                }
                else
                    return true;
            }
            return false;
        };

        int outRawId = 0;
        auto updateNdimOutId = [&]()
        {
            std::vector<int> nDimOutId(outShape[0].size(), 0);
            //TODO optimize
            for (int i = 0; i < nInpDims; i++)
            {
                if (dimsToKeep[i])
                {
                    if (dimsToReduce[i])
                    {
                        nDimOutId.push_back(0);
                    }
                    else
                    {
                        nDimOutId.push_back(nDimInpId[i]);
                    }
                }
                else
                {
                    CV_Assert(dimsToReduce[i]);
                }
            }
            //TODO: use cv index convertor
            long factor = 1;
            outRawId = 0;
            for (int i = nDimOutId.size(); i >= 0; i--)
            {
                outRawId += nDimOutId[i] * factor;
                factor *= outShape[0][i];
            }
            return true;
        };

        for (int i = 0; i < inputs[0].total(); i++)
        {
            const auto& inpElem = inputs[0].at<float>(i);
            auto& outElem = mOut.at<float>(outRawId);
            if (type == ReduceType::L1)
            {
                outElem += inpElem;
            }
            else
            {
                outElem += inpElem * inpElem;
            }

            increaseNdimId();
            updateNdimOutId();
        };

        if (type == ReduceType::L2)
        {
            for (auto i = mOut.begin<float>(); i != mOut.end<float>(); i++)
            {
                *i = std::sqrt(*i);
            }
        }
    }

private:
    // The axis in which to compute the arg indices. Accepted range is [-r, r-1] where r = rank(data).
    DictValue axes;
    // Keep dimensions
    bool keepDims;
    // Type of reduce layer L1/L2
    enum class ReduceType
    {
        L1 = 0,
        L2 = 1,
    };
    ReduceType type;
};

Ptr<ReduceLayer> ReduceLayer::create(const LayerParams& params)
{
    return Ptr<ReduceLayerImpl>(new ReduceLayerImpl(params));
}

}}  // namespace cv::dnn
