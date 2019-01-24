// Test already trained model

#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <iostream>
#include <string>

#include "PAIGE_Loss.h"
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace shark;


int main(int argc,char ** argv) {

    if(argc!=3)
    {
        cout<<"Usage:"<<endl;
        cout<<"~/TestModel /path/to/model /path/to/test/dataset "<<endl
            <<"eg. ~/TestModel /home/ML_Models/svm.model /home/Data/data.csv"<<endl;
        return EXIT_FAILURE;
    }


//    string training_data_path;
    string test_data_path;
    string model_name;

//    training_data_path=argv[1];
    test_data_path=argv[2];
    model_name=argv[1];
//    model_name=model_name+".model";

//    ClassificationDataset data;
//    importCSV(data,training_data_path.c_str(),FIRST_COLUMN);



    //Split the dataset into a training and a test dataset
    ClassificationDataset dataTest ;
    importCSV(dataTest,test_data_path.c_str(),FIRST_COLUMN);
    //###end<import>

//    cout << "Training set - number of data points: " << data.numberOfElements()
//         << " number of classes: " << numberOfClasses(data)
//         << " input dimension: " << inputDimension(data) << endl;
//
    cout << "Test set - number of data points: " << dataTest.numberOfElements()
         << " number of classes: " << numberOfClasses(dataTest)
         << " input dimension: " << inputDimension(dataTest) << endl;

    //Generate a random forest
    //###begin<train>

//    data.append(dataTest);

//    RFTrainer trainer;
//    RFClassifier model;
//
//    trainer.setNTrees(50);
//    trainer.setNodeSize(3);
//
//    trainer.train(model, data);
    //###end<train>

//    unsigned int cnt{0};
//    Data<unsigned int > labels=data.labels();
//    unsigned int number_of_0{0},number_of_1{0};
//    double ratio;
//    for(size_t i=0;i<labels.numberOfElements();++i)
//    {
//        ++cnt;
//        if(labels.element(i)==0)
//        {
//            ++number_of_0;
//        }
//        else if(labels.element(i)==1)
//        {
//            ++number_of_1;
//        }
//        else
//        {
//            cout<<"Error!"<<endl;
//        }
//    }
//
//    ratio=(double)number_of_0/(double)number_of_1;
//    cout<<"Number of 0:"<<number_of_0<<endl
//        <<"Number of 1:"<<number_of_1<<endl
//        <<"Ratio:"<<ratio<<endl
//        <<"Total data points:"<<cnt<<endl;
//
//    ofstream ofs(model_name.c_str());
//    boost::archive::text_oarchive oa(ofs);
//    model.write(oa);
//    ofs.close();


//    std::string model_name="/home/xjxj/data/PAIGE_dataset_total/PAIGE_test_data/PAIGE_unknown/Dante_unknown.model";
    RFClassifier new_model;
    ifstream ifs(model_name);
    boost::archive::text_iarchive ia(ifs);
    new_model.read(ia);
    ifs.close();
//
//
//    RealVector v(200);
//    for(int i=0;i<200;++i)
//    {
//        v(i)=0.05;
//    }
//
////    Predict single PAIGE feature
//    RealVector prediction_for_test=new_model(v);
//    cout<<"Prediction for test"<<endl;
//    cout<<prediction_for_test<<endl;
//    cout<<prediction_for_test.size()<<endl;

    // evaluate Random Forest classifier
    //###begin<eval>
//    ZeroOneLoss< unsigned int,RealVector > loss;
//    Data<RealVector> prediction = new_model(data.inputs());
//    Data<unsigned int> labels=data.labels();
//    cout << "Random Forest on training set accuracy: " << 1. - loss.eval(labels, prediction) << endl;
//

    Data<RealVector> prediction;
    Data<unsigned int> labels;
//    ZeroOneLoss<unsigned int,RealVector> loss;
//
    labels=dataTest.labels();
    prediction = new_model(dataTest.inputs());
//    cout << "Random Forest on test set accuracy:     " << 1. - loss.eval(labels, prediction) << endl;
//###end<eval>


    PAIGE_Loss paige_loss;
    paige_loss.evalScore(labels,prediction);

    double accuracy,precision,recall;
    accuracy=paige_loss.getAccuracy();
    precision=paige_loss.getPrecision();
    recall=paige_loss.getRecall();

    cout<<endl<<"PAIGE_Loss Evaluation:"<<endl;
    cout<<"Accuracy:"<<accuracy<<endl;
    cout<<"Precision:"<<precision<<endl;
    cout<<"Recall:"<<recall<<endl;
    cout<<"TP:"<<paige_loss.getTP()<<endl;
    cout<<"FP:"<<paige_loss.getFP()<<endl;
    cout<<"TN:"<<paige_loss.getTN()<<endl;
    cout<<"FN:"<<paige_loss.getFN()<<endl;

    return EXIT_SUCCESS;
}