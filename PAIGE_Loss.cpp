//
// Created by xjxj on 19-1-15.
//

#include "PAIGE_Loss.h"
#include <iostream>

void PAIGE_Loss::evalScore(const Data<unsigned int> & labels, const Data<RealVector> & predictions )
{
    SIZE_CHECK(predictions.numberOfElements()!=0);
    SIZE_CHECK(predictions.numberOfBatches()==labels.numberOfBatches());
    SIZE_CHECK(predictions.numberOfElements()==labels.numberOfElements());

    _TP=0;
    _FP=0;
    _TN=0;
    _FN=0;

    unsigned int current_label{0};
    RealVector current_prediction;

    for(std::size_t i=0;i!=predictions.numberOfElements();++i)
    {
        current_label=labels.element(i);
        current_prediction=predictions.element(i);

        std::size_t size=current_prediction.size();
        RANGE_CHECK(current_label < size);

        if(current_label==1)
        {
            if(current_prediction(0)>current_prediction(1))
                ++_FN;
            else
                ++_TP;
        }
        else
        {
            if(current_prediction(0)>current_prediction(1))
                ++_TN;
            else
                ++_FP;
        }
    }

    _accuracy=(double)(_TP+_TN)/predictions.numberOfElements();
    _precision=(double)_TP/(_TP+_FP);
    _recall=(double)_TP/(_TP+_FN);
}

int PAIGE_Loss::getTP() const
{
    return _TP;
}

int PAIGE_Loss::getFP() const
{
    return _FP;
}

int PAIGE_Loss::getTN() const
{
    return _TN;
}

int PAIGE_Loss::getFN() const
{
    return _FN;
}

double PAIGE_Loss::getAccuracy() const
{
    return _accuracy;
}

double PAIGE_Loss::getPrecision() const
{
    return _precision;
}

double PAIGE_Loss::getRecall() const
{
    return _recall;
}

