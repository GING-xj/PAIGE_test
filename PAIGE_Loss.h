//
// Created by xjxj on 19-1-15.
//

#ifndef PAIGE_PAIGE_LOSS_H
#define PAIGE_PAIGE_LOSS_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

using namespace shark;

class PAIGE_Loss {

public:
    PAIGE_Loss():_TP(0),_FP(0),_TN(0),_FN(0),_accuracy(0),_precision(0),_recall(0) { }
    ~PAIGE_Loss() { }

    void evalScore(const Data<unsigned int> & labels, const Data<RealVector> & predictions );

    int getTP() const;

    int getFP() const;

    int getTN() const;

    int getFN() const;

    double getAccuracy() const;

    double getPrecision() const;

    double getRecall() const;


private:
    int _TP,_FP,_TN,_FN;
    double _accuracy,_precision,_recall;
    const int _num_of_class=2;
};


#endif //PAIGE_PAIGE_LOSS_H
